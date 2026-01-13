from datetime import date
from typing import List, Optional
from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    BigInteger, Text, Integer, Date, TIMESTAMP, ForeignKey, func, text
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql import select


# Flask app initialization
app = Flask(__name__)

# Swagger UI configuration
SWAGGER_URL = '/api/docs'
API_URL = '/api/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Action Plan API"
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Database configuration
DATABASE_URL = "postgresql+psycopg2://administrationSTS:St%24%400987@avo-adb-002.postgres.database.azure.com:5432/ActionPlan_Supplier?sslmode=require"
engine = create_engine(DATABASE_URL, future=True)
metadata = MetaData(schema="public")

# database pour meeting 
DATABASE_URL_MEETING = "postgresql+psycopg2://.../MeetingDB?sslmode=require"
engine_meeting = create_engine(DATABASE_URL_MEETING, future=True)

# Define database tables
sujet = Table(
    "sujet", metadata,
    Column("id", BigInteger, primary_key=True),
    Column("code", Text, unique=True),
    Column("titre", Text, nullable=False),
    Column("description", Text),
    Column("created_at", TIMESTAMP(timezone=True), server_default=func.now()),
    Column("updated_at", TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now()),
    Column("parent_sujet_id", BigInteger, ForeignKey("public.sujet.id", onupdate="CASCADE", ondelete="SET NULL"))
)

action = Table(
    "action", metadata,
    Column("id", BigInteger, primary_key=True),
    Column("sujet_id", BigInteger, ForeignKey("public.sujet.id", ondelete="CASCADE"), nullable=False),
    Column("parent_action_id", BigInteger, ForeignKey("public.action.id", ondelete="CASCADE")),
    Column("type", Text, nullable=False),
    Column("titre", Text, nullable=False),
    Column("description", Text),
    Column("status", Text, server_default=text("'open'")),
    Column("priorite", Integer),
    Column("responsable", Text),
    Column("due_date", Date),
    Column("ordre", Integer),
    Column("depth", Integer),
    Column("created_at", TIMESTAMP(timezone=True), server_default=func.now()),
    Column("updated_at", TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now()),
)

# --------------------------------------------
# Pydantic models (validation)
# --------------------------------------------
VALID_STATUSES = {"open", "closed", "blocked"}

class ActionNode(BaseModel):
    titre: str = Field(..., min_length=1)
    description: Optional[str] = None
    responsable: Optional[str] = None
    priorite: Optional[int] = Field(None, ge=0)
    due_date: Optional[date] = None
    status: Optional[str] = Field("open")
    sous_actions: List["ActionNode"] = Field(default_factory=list)

    @field_validator("status")
    def status_must_be_valid(cls, v):
        if v is None:
            return "open"
        if v not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
        return v

ActionNode.model_rebuild()

class SujetNode(BaseModel):
    titre: str = Field(..., min_length=1)
    code: Optional[str] = None
    description: Optional[str] = None
    sous_sujets: List["SujetNode"] = Field(default_factory=list)
    actions: List[ActionNode] = Field(default_factory=list)

SujetNode.model_rebuild()

class PlanV1(BaseModel):
    version: str = Field(..., pattern=r"^1\.0$")
    plan_code: Optional[str] = None
    plan_title: str = Field(..., min_length=1)
    sujets: List[SujetNode] = Field(default_factory=list)

# --------------------------------------------
# DB helper functions
# --------------------------------------------
def upsert_sujet(conn: Connection,
                 titre: str,
                 parent_sujet_id: Optional[int],
                 code: Optional[str],
                 description: Optional[str]) -> int:
    """
    Upsert a subject by code (if provided) or by (parent_sujet_id, titre).
    Returns the subject ID.
    """
    if code:
        stmt = pg_insert(sujet).values(
            code=code, titre=titre, description=description, parent_sujet_id=parent_sujet_id
        ).on_conflict_do_update(
            index_elements=["code"],
            set_=dict(
                titre=titre,
                description=description,
                parent_sujet_id=parent_sujet_id,
                updated_at=func.now()
            )
        ).returning(sujet.c.id)
        return conn.execute(stmt).scalar_one()
    else:
        # No code: use SELECT-then-UPDATE/INSERT
        if parent_sujet_id is None:
            sel_stmt = select(sujet.c.id).where(
                sujet.c.parent_sujet_id.is_(None),
                sujet.c.titre == titre
            )
        else:
            sel_stmt = select(sujet.c.id).where(
                sujet.c.parent_sujet_id == parent_sujet_id,
                sujet.c.titre == titre
            )
        
        existing_id_row = conn.execute(sel_stmt).first()
        
        if existing_id_row:
            existing_id = existing_id_row[0]
            upd_stmt = sujet.update().where(
                sujet.c.id == existing_id
            ).values(
                description=description,
                updated_at=func.now()
            ).returning(sujet.c.id)
            return conn.execute(upd_stmt).scalar_one()
        else:
            ins_stmt = sujet.insert().values(
                titre=titre,
                description=description,
                parent_sujet_id=parent_sujet_id
            ).returning(sujet.c.id)
            return conn.execute(ins_stmt).scalar_one()

def insert_action_recursive(conn: Connection,
                             sujet_id: int,
                             parent_action_id: Optional[int],
                             node: ActionNode,
                             ordre: int = 0) -> int:
    """
    Insert an action and all its sub-actions recursively.
    Returns the action ID.
    """
    def level_type(level: int) -> str:
        if level <= 0: 
            return "action"
        if level == 1: 
            return "sub_action"
        return "sub_sub_action"

    act_level = 0
    if parent_action_id:
        parent = conn.execute(
            select(action.c.depth).where(action.c.id == parent_action_id)
        ).first()
        if parent and parent[0] is not None:
            act_level = min(int(parent[0]) + 1, 2)
        else:
            act_level = 1

    row = conn.execute(
        action.insert().values(
            sujet_id=sujet_id,
            parent_action_id=parent_action_id,
            type=level_type(act_level),
            titre=node.titre,
            description=node.description,
            status=node.status or "open",
            priorite=node.priorite,
            responsable=node.responsable,
            due_date=node.due_date,
            ordre=ordre
        ).returning(action.c.id)
    ).first()
    new_id = int(row[0])

    # Recurse for sub-actions
    for idx, child in enumerate(node.sous_actions):
        insert_action_recursive(conn, sujet_id, new_id, child, idx)

    return new_id

def ingest_sujet_tree(conn: Connection, node: SujetNode, parent_id: Optional[int]) -> int:
    """
    Recursively ingest a subject tree with actions.
    Returns the subject ID.
    """
    this_id = upsert_sujet(conn,
                           titre=node.titre,
                           parent_sujet_id=parent_id,
                           code=node.code,
                           description=node.description)

    # Insert actions under this subject
    for idx, a in enumerate(node.actions):
        insert_action_recursive(conn, sujet_id=this_id, parent_action_id=None, node=a, ordre=idx)

    # Insert nested subjects
    for s in node.sous_sujets:
        ingest_sujet_tree(conn, s, this_id)

    return this_id

def ingest_plan(conn: Connection, plan: PlanV1) -> int:
    """
    Create/update a root subject for the plan and ingest all subjects/actions.
    Returns the root subject ID.
    """
    root_code = plan.plan_code
    root_titre = plan.plan_title
    root_desc = "Action plan root (ingested by assistant)"

    root_id = upsert_sujet(conn,
                           titre=root_titre,
                           parent_sujet_id=None,
                           code=root_code,
                           description=root_desc)

    for s in plan.sujets:
        ingest_sujet_tree(conn, s, root_id)

    return root_id

# --------------------------------------------
# Flask routes
# --------------------------------------------

@app.route("/api/swagger.json", methods=["GET"])
def swagger_spec():
    """OpenAPI specification endpoint"""
    return jsonify({
        "openapi": "3.0.0",
        "info": {
            "title": "Action Plan API",
            "description": "API for managing hierarchical action plans with subjects and actions",
            "version": "1.0.0"
        },
        "servers": [
            {
                "url": "https://action-plan-db.azurewebsites.net",
                "description": "Development server"
            }
        ],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check",
                    "description": "Check if the API is running",
                    "responses": {
                        "200": {
                            "description": "API is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "ok": {"type": "boolean"},
                                            "status": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/schema": {
                "get": {
                    "summary": "Get example schema",
                    "description": "Returns an example JSON structure for action plans",
                    "responses": {
                        "200": {
                            "description": "Example schema",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/PlanV1"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/plans": {
                "post": {
                    "summary": "Create/Update action plan",
                    "description": "Ingest a complete action plan with subjects and actions",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/PlanV1"
                                },
                                "example": {
                                    "version": "1.0",
                                    "plan_code": "AP-2025-10-OPS-001",
                                    "plan_title": "Q4 Operations Readiness",
                                    "sujets": [
                                        {
                                            "titre": "Maintenance",
                                            "code": "OPS-MNT",
                                            "description": "Preventive maintenance",
                                            "sous_sujets": [],
                                            "actions": [
                                                {
                                                    "titre": "Create PM checklist",
                                                    "description": "Draft checklist",
                                                    "responsable": "jane.doe",
                                                    "priorite": 2,
                                                    "due_date": "2025-11-15",
                                                    "status": "open",
                                                    "sous_actions": []
                                                }
                                            ]
                                        }
                                    ]
                                }
                            }
                        }
                    },
                    "responses": {
                        "201": {
                            "description": "Plan created successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "root_sujet_id": {
                                                "type": "integer",
                                                "description": "ID of the root subject"
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Validation error",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Error"
                                    }
                                }
                            }
                        },
                        "409": {
                            "description": "Database integrity error",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Error"
                                    }
                                }
                            }
                        },
                        "500": {
                            "description": "Server error",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Error"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "PlanV1": {
                    "type": "object",
                    "required": ["version", "plan_title", "sujets"],
                    "properties": {
                        "version": {
                            "type": "string",
                            "pattern": "^1\\.0$",
                            "description": "Schema version (must be 1.0)"
                        },
                        "plan_code": {
                            "type": "string",
                            "nullable": True,
                            "description": "Unique plan code"
                        },
                        "plan_title": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Plan title"
                        },
                        "sujets": {
                            "type": "array",
                            "items": {
                                "$ref": "#/components/schemas/SujetNode"
                            }
                        }
                    }
                },
                "SujetNode": {
                    "type": "object",
                    "required": ["titre"],
                    "properties": {
                        "titre": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Subject title"
                        },
                        "code": {
                            "type": "string",
                            "nullable": True,
                            "description": "Unique subject code"
                        },
                        "description": {
                            "type": "string",
                            "nullable": True,
                            "description": "Subject description"
                        },
                        "sous_sujets": {
                            "type": "array",
                            "items": {
                                "$ref": "#/components/schemas/SujetNode"
                            },
                            "description": "Nested sub-subjects"
                        },
                        "actions": {
                            "type": "array",
                            "items": {
                                "$ref": "#/components/schemas/ActionNode"
                            },
                            "description": "Actions under this subject"
                        }
                    }
                },
                "ActionNode": {
                    "type": "object",
                    "required": ["titre"],
                    "properties": {
                        "titre": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Action title"
                        },
                        "description": {
                            "type": "string",
                            "nullable": True,
                            "description": "Action description"
                        },
                        "responsable": {
                            "type": "string",
                            "nullable": True,
                            "description": "Person responsible"
                        },
                        "priorite": {
                            "type": "integer",
                            "minimum": 0,
                            "nullable": True,
                            "description": "Priority level"
                        },
                        "due_date": {
                            "type": "string",
                            "format": "date",
                            "nullable": True,
                            "description": "Due date (YYYY-MM-DD)"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["open", "closed", "blocked"],
                            "default": "open",
                            "description": "Action status"
                        },
                        "sous_actions": {
                            "type": "array",
                            "items": {
                                "$ref": "#/components/schemas/ActionNode"
                            },
                            "description": "Nested sub-actions"
                        }
                    }
                },
                "Error": {
                    "type": "object",
                    "properties": {
                        "error": {
                            "type": "string",
                            "description": "Error type"
                        },
                        "detail": {
                            "type": "string",
                            "description": "Error details"
                        }
                    }
                }
            }
        }
    })

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"ok": True, "status": "running"})

@app.route("/api/plans", methods=["POST"])
def post_plan():
    """
    POST endpoint to ingest an action plan.
    Body: JSON matching PlanV1 schema
    Returns: { root_sujet_id: <id> }
    """
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": "invalid_json", "detail": str(e)}), 400
    
    try:
        plan = PlanV1.model_validate(data)
    except Exception as e:
        return jsonify({"error": "validation_error", "detail": str(e)}), 400

    try:
        with engine.begin() as conn:
            root_id = ingest_plan(conn, plan)
            return jsonify({"root_sujet_id": root_id}), 201
    except IntegrityError as ie:
        return jsonify({"error": "db_integrity_error", "detail": str(ie.orig)}), 409
    except Exception as e:
        return jsonify({"error": "server_error", "detail": str(e)}), 500
        
@app.route("/api/plans-meeting", methods=["POST"])
def post_plan_meeting():
    """
    POST endpoint to ingest an action plan into Meeting DB.
    Body: JSON matching PlanV1 schema
    Returns: { root_sujet_id: <id> }
    """
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": "invalid_json", "detail": str(e)}), 400

    try:
        plan = PlanV1.model_validate(data)
    except Exception as e:
        return jsonify({"error": "validation_error", "detail": str(e)}), 400

    try:
        with engine_meeting.begin() as conn:
            root_id = ingest_plan(conn, plan)
            return jsonify({"root_sujet_id": root_id}), 201
    except IntegrityError as ie:
        return jsonify({"error": "db_integrity_error", "detail": str(ie.orig)}), 409
    except Exception as e:
        return jsonify({"error": "server_error", "detail": str(e)}), 500


@app.route("/api/schema", methods=["GET"])
def get_schema():
    """
    Returns example JSON schema for action plans (v1.0)
    """
    return jsonify({
        "version": "1.0",
        "plan_code": "AP-2025-10-OPS-001",
        "plan_title": "Q4 Operations Readiness",
        "sujets": [
            {
                "titre": "Maintenance",
                "code": "OPS-MNT",
                "description": "Preventive and corrective maintenance plan.",
                "sous_sujets": [
                    {
                        "titre": "Compressors",
                        "description": "Air compressor reliability",
                        "sous_sujets": [],
                        "actions": []
                    }
                ],
                "actions": [
                    {
                        "titre": "Create weekly PM checklist",
                        "description": "Draft and validate PM checklist with production.",
                        "responsable": "jane.doe",
                        "priorite": 2,
                        "due_date": "2025-11-15",
                        "status": "open",
                        "sous_actions": [
                            {
                                "titre": "Collect OEM manuals",
                                "due_date": "2025-10-31",
                                "sous_actions": [
                                    {
                                        "titre": "Request missing manuals from supplier",
                                        "due_date": "2025-10-27",
                                        "sous_actions": []
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "not_found", "detail": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({"error": "internal_server_error", "detail": str(error)}), 500

# --------------------------------------------
# Run the application
# --------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Action Plan API Starting...")
    print("="*60)
    print(f"📋 Swagger UI: http://localhost:5000{SWAGGER_URL}")
    print(f"📄 API Docs: http://localhost:5000{API_URL}")
    print(f"❤️  Health Check: http://localhost:5000/health")
    print("="*60 + "\n")
    
    # Run Flask development server
    app.run(host="0.0.0.0", port=5000, debug=True)
