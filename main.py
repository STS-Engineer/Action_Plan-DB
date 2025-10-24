from datetime import date
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    create_engine, MetaData, Table, Column,
    BigInteger, Text, Integer, Date, TIMESTAMP, ForeignKey, func, text
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql import select
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from db import get_connection

# ---------------------------------------------------------------------------------------
# Configuration de la base de données
# ---------------------------------------------------------------------------------------
DATABASE_URL = "postgresql+psycopg2://administrationSTS:St%24%400987@avo-adb-002.postgres.database.azure.com:5432/Action Plan?sslmode=require"
engine = create_engine(DATABASE_URL, future=True)
metadata = MetaData(schema="public")

# Définition des tables
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

# ---------------------------------------------------------------------------------------
# Modèles Pydantic
# ---------------------------------------------------------------------------------------
VALID_STATUSES = {"open", "closed", "blocked"}

class ActionNode(BaseModel):
    titre: str = Field(..., min_length=1)
    description: Optional[str] = None
    responsable: Optional[str] = None
    priorite: Optional[int] = Field(None, ge=0)
    due_date: Optional[date] = None
    status: Optional[str] = Field(default="open")
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

# ---------------------------------------------------------------------------------------
# Fonctions helper pour la base de données
# ---------------------------------------------------------------------------------------
def upsert_sujet(conn: Connection,
                 titre: str,
                 parent_sujet_id: Optional[int],
                 code: Optional[str],
                 description: Optional[str]) -> int:
    """
    Upsert d'un sujet par code (si fourni) ou par (parent_sujet_id, titre)
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
        # Pas de code fourni: SELECT-then-UPDATE/INSERT
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
            # Mise à jour
            existing_id = existing_id_row[0]
            upd_stmt = sujet.update().where(
                sujet.c.id == existing_id
            ).values(
                description=description,
                updated_at=func.now()
            ).returning(sujet.c.id)
            return conn.execute(upd_stmt).scalar_one()
        else:
            # Insertion
            ins_stmt = sujet.insert().values(
                titre=titre,
                description=description,
                parent_sujet_id=parent_sujet_id
            ).returning(sujet.c.id)
            return conn.execute(ins_stmt).scalar_one()

def insert_action_recursive(conn: Connection,
                             sujet_id: int,
                             parent_action_id: Optional[int],
                             node: ActionNode) -> int:
    """
    Insertion récursive d'une action et de ses sous-actions
    """
    def level_type(level: int) -> str:
        if level <= 0: return "action"
        if level == 1: return "sub_action"
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
            ordre=None
        ).returning(action.c.id)
    ).first()
    new_id = int(row[0])

    # Récursion sur les sous-actions
    for child in node.sous_actions:
        insert_action_recursive(conn, sujet_id, new_id, child)

    return new_id

def ingest_sujet_tree(conn: Connection, node: SujetNode, parent_id: Optional[int]) -> int:
    """
    Ingestion récursive d'un arbre de sujets
    """
    this_id = upsert_sujet(conn,
                           titre=node.titre,
                           parent_sujet_id=parent_id,
                           code=node.code,
                           description=node.description)

    # Actions directement sous ce sujet
    for a in node.actions:
        insert_action_recursive(conn, sujet_id=this_id, parent_action_id=None, node=a)

    # Sujets imbriqués
    for s in node.sous_sujets:
        ingest_sujet_tree(conn, s, this_id)

    return this_id

def ingest_plan(conn: Connection, plan: PlanV1) -> int:
    """
    Crée/met à jour un sujet racine pour le plan et ingère tous les sujets/actions
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

# ---------------------------------------------------------------------------------------
# Application FastAPI
# ---------------------------------------------------------------------------------------
app = FastAPI(
    title="Plans d'Action API",
    description="API pour la gestion des plans d'action avec sujets et actions hiérarchiques",
    version="1.0"
)

@app.get("/health")
def health():
    """Point de contrôle de santé de l'API"""
    return {"ok": True}

@app.post("/api/plans")
def post_plan(plan: PlanV1):
    """
    Créer ou mettre à jour un plan d'action complet
    
    - **plan**: Structure JSON conforme au schéma PlanV1
    - **Returns**: ID du sujet racine créé
    """
    with engine.begin() as conn:
        try:
            root_id = ingest_plan(conn, plan)
            return {"root_sujet_id": root_id}
        except IntegrityError as ie:
            raise HTTPException(
                status_code=409,
                detail={"error": "db_integrity_error", "detail": str(ie.orig)}
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={"error": "server_error", "detail": str(e)}
            )

@app.get("/api/schema")
def get_schema():
    """
    Retourne un exemple de schéma JSON pour créer un plan d'action
    """
    return {
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
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
