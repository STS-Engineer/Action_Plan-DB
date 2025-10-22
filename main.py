# mai.py
from fastapi import FastAPI, HTTPException, Query, Depends
from sqlalchemy import (
    create_engine, Column, BigInteger, Text, Integer, Date, DateTime, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import func
from pydantic import BaseModel
from typing import Optional, List
from datetime import date, datetime

# <<< Import de la connexion centralisée >>>
from db import get_connection

# ========================================
# CONFIGURATION DATABASE
# ========================================

# On délègue l'ouverture de chaque connexion DB à db.get_connection()
engine = create_engine(
    "postgresql+psycopg2://",  # DSN vide car on fournit creator=
    creator=get_connection,
    pool_pre_ping=True,        # évite les connexions mortes
    future=True                # API SQLAlchemy 2.x
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
Base = declarative_base()

app = FastAPI(title="Action Plan API", version="1.0.0")


# -------------------------
# Dépendance DB FastAPI
# -------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ========================================
# MODÈLES SQLAlchemy
# ========================================

class SujetModel(Base):
    __tablename__ = 'sujet'

    id = Column(BigInteger, primary_key=True, index=True)
    code = Column(Text, unique=True)
    titre = Column(Text, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    parent_sujet_id = Column(BigInteger, ForeignKey('sujet.id', ondelete='CASCADE'), index=True)

    # Relations (self-referential)
    parent = relationship(
        'SujetModel',
        remote_side=[id],
        back_populates='enfants',
        uselist=False,
    )
    enfants = relationship(
        'SujetModel',
        back_populates='parent',
        cascade='all, delete-orphan',
        single_parent=True,          # <-- requis avec delete-orphan
        passive_deletes=True,        # optionnel si tu comptes sur ON DELETE CASCADE côté DB
        lazy='selectin'              # optionnel: évite N+1
    )

    actions = relationship(
        'ActionModel',
        back_populates='sujet',
        cascade='all, delete-orphan',
        passive_deletes=True,
        lazy='selectin'
    )


class ActionModel(Base):
    __tablename__ = 'action'

    id = Column(BigInteger, primary_key=True, index=True)
    sujet_id = Column(BigInteger, ForeignKey('sujet.id', ondelete='CASCADE'), nullable=False, index=True)
    parent_action_id = Column(BigInteger, ForeignKey('action.id', ondelete='CASCADE'), index=True)
    type = Column(Text, nullable=False)
    titre = Column(Text, nullable=False)
    description = Column(Text)
    status = Column(Text, default='nouveau')
    priorite = Column(Integer)
    responsable = Column(Text)
    due_date = Column(Date)
    ordre = Column(Integer)
    depth = Column(Integer)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    # Relations
    sujet = relationship('SujetModel', back_populates='actions')

    parent_action = relationship(
        'ActionModel',
        remote_side=[id],
        back_populates='sous_actions',
        uselist=False,
    )
    sous_actions = relationship(
        'ActionModel',
        back_populates='parent_action',
        cascade='all, delete-orphan',
        single_parent=True,          # <-- idem
        passive_deletes=True,
        lazy='selectin'
    )

# ========================================
# SCHÉMAS PYDANTIC (compat v1/v2)
# ========================================

# Compat helper: v2 => .model_validate, v1 => .from_orm
def to_schema(schema_cls, obj):
    if hasattr(schema_cls, "model_validate"):
        return schema_cls.model_validate(obj)
    return schema_cls.from_orm(obj)


def set_from_attributes(cls):
    # pydantic v2
    try:
        from pydantic import ConfigDict  # type: ignore
        cls.model_config = ConfigDict(from_attributes=True)  # type: ignore
    except Exception:
        # pydantic v1
        class Config:
            orm_mode = True
        cls.Config = Config
    return cls


@set_from_attributes
class SujetBase(BaseModel):
    code: Optional[str] = None
    titre: str
    description: Optional[str] = None
    parent_sujet_id: Optional[int] = None


class SujetCreate(SujetBase):
    pass


class SujetUpdate(BaseModel):
    code: Optional[str] = None
    titre: Optional[str] = None
    description: Optional[str] = None
    parent_sujet_id: Optional[int] = None


@set_from_attributes
class ActionBase(BaseModel):
    sujet_id: int
    parent_action_id: Optional[int] = None
    type: str
    titre: str
    description: Optional[str] = None
    status: str = 'nouveau'
    priorite: Optional[int] = None
    responsable: Optional[str] = None
    due_date: Optional[date] = None
    ordre: Optional[int] = None


class ActionCreate(ActionBase):
    pass


class ActionUpdate(BaseModel):
    sujet_id: Optional[int] = None
    parent_action_id: Optional[int] = None
    type: Optional[str] = None
    titre: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    priorite: Optional[int] = None
    responsable: Optional[str] = None
    due_date: Optional[date] = None
    ordre: Optional[int] = None


@set_from_attributes
class ActionResponse(ActionBase):
    id: int
    depth: Optional[int] = None
    created_at: Optional[datetime] = None   # <- était Optional[str]
    updated_at: Optional[datetime] = None   # <- était Optional[str]
    sous_actions: Optional[List['ActionResponse']] = None


@set_from_attributes
class SujetResponse(SujetBase):
    id: int
    created_at: Optional[datetime] = None   # <- était Optional[str]
    updated_at: Optional[datetime] = None   # <- était Optional[str]
    enfants: Optional[List['SujetResponse']] = None
    actions: Optional[List[ActionResponse]] = None


class SujetArborescenceResponse(BaseModel):
    chemin_parents: List[SujetResponse]
    sujet_actuel: SujetResponse


class ActionArborescenceResponse(BaseModel):
    sujet: SujetResponse
    chemin_parents: List[ActionResponse]
    action_actuelle: ActionResponse


# Forward refs (v1/v2)
try:
    SujetResponse.model_rebuild()  # pydantic v2
    ActionResponse.model_rebuild()
except Exception:
    SujetResponse.update_forward_refs()  # pydantic v1
    ActionResponse.update_forward_refs()


# ========================================
# FONCTIONS UTILITAIRES
# ========================================

def sujet_to_dict(sujet: SujetModel, include_children: bool = False, include_actions: bool = False) -> SujetResponse:
    data = to_schema(SujetResponse, sujet)

    if include_children and sujet.enfants:
        data.enfants = [sujet_to_dict(e, include_children=True) for e in sujet.enfants]

    if include_actions and sujet.actions:
        data.actions = [action_to_dict(a, include_children=True) for a in sujet.actions]

    return data


def action_depth(action: ActionModel) -> int:
    """Calcule la profondeur réelle en remontant la chaîne parent -> ..."""
    depth = 0
    current = action.parent_action
    while current is not None:
        depth += 1
        current = current.parent_action
    return depth


def action_to_dict(action: ActionModel, include_children: bool = False) -> ActionResponse:
    data = to_schema(ActionResponse, action)
    if include_children and action.sous_actions:
        data.sous_actions = [action_to_dict(sa, include_children=True) for sa in action.sous_actions]
    return data


# ========================================
# APIs SUJETS
# ========================================

@app.post("/api/sujets", response_model=SujetResponse, status_code=201)
def create_sujet(sujet: SujetCreate, db: Session = Depends(get_db)):
    """Créer un nouveau sujet (peut être un sous-sujet si parent_sujet_id est fourni)"""
    # Vérifier si le parent existe (si fourni)
    if sujet.parent_sujet_id:
        parent = db.query(SujetModel).filter(SujetModel.id == sujet.parent_sujet_id).first()
        if not parent:
            raise HTTPException(status_code=404, detail="Sujet parent introuvable")

    db_sujet = SujetModel(**sujet.dict())
    db.add(db_sujet)
    db.flush()
    db.refresh(db_sujet)
    db.commit()

    return sujet_to_dict(db_sujet)


@app.get("/api/sujets/{sujet_id}", response_model=SujetResponse)
def get_sujet(
    sujet_id: int,
    include_children: bool = Query(False, description="Inclure tous les enfants récursivement"),
    include_actions: bool = Query(False, description="Inclure toutes les actions"),
    db: Session = Depends(get_db),
):
    """Récupérer un sujet avec option d'inclure toute sa hiérarchie"""
    sujet = db.query(SujetModel).filter(SujetModel.id == sujet_id).first()
    if not sujet:
        raise HTTPException(status_code=404, detail="Sujet introuvable")

    return sujet_to_dict(sujet, include_children=include_children, include_actions=include_actions)


@app.get("/api/sujets", response_model=dict)
def get_all_sujets(
    include_hierarchy: bool = Query(False, description="Inclure toute la hiérarchie des enfants"),
    db: Session = Depends(get_db),
):
    """Récupérer tous les sujets racines (sans parent) avec leur hiérarchie complète"""
    sujets_racines = db.query(SujetModel).filter(SujetModel.parent_sujet_id == None).all()

    return {
        'count': len(sujets_racines),
        'sujets': [sujet_to_dict(s, include_children=include_hierarchy) for s in sujets_racines]
    }


@app.get("/api/sujets/{sujet_id}/enfants", response_model=dict)
def get_sujet_enfants(sujet_id: int, db: Session = Depends(get_db)):
    """Récupérer tous les enfants directs d'un sujet"""
    sujet = db.query(SujetModel).filter(SujetModel.id == sujet_id).first()
    if not sujet:
        raise HTTPException(status_code=404, detail="Sujet introuvable")

    return {
        'sujet_id': sujet_id,
        'titre': sujet.titre,
        'enfants': [sujet_to_dict(e) for e in sujet.enfants]
    }


@app.get("/api/sujets/{sujet_id}/arborescence", response_model=SujetArborescenceResponse)
def get_sujet_arborescence(sujet_id: int, db: Session = Depends(get_db)):
    """Récupérer l'arborescence complète (parents et enfants) d'un sujet"""
    sujet = db.query(SujetModel).filter(SujetModel.id == sujet_id).first()
    if not sujet:
        raise HTTPException(status_code=404, detail="Sujet introuvable")

    # Remonter jusqu'à la racine
    chemin_parents: List[SujetResponse] = []
    current = sujet
    while getattr(current, "parent", None):
        chemin_parents.insert(0, sujet_to_dict(current.parent))
        current = current.parent

    return {
        'chemin_parents': chemin_parents,
        'sujet_actuel': sujet_to_dict(sujet, include_children=True)
    }


@app.put("/api/sujets/{sujet_id}", response_model=SujetResponse)
def update_sujet(sujet_id: int, sujet_update: SujetUpdate, db: Session = Depends(get_db)):
    """Mettre à jour un sujet existant"""
    sujet = db.query(SujetModel).filter(SujetModel.id == sujet_id).first()
    if not sujet:
        raise HTTPException(status_code=404, detail="Sujet introuvable")

    # Vérifier si le nouveau parent existe (si fourni)
    if sujet_update.parent_sujet_id is not None:
        if sujet_update.parent_sujet_id == sujet_id:
            raise HTTPException(status_code=400, detail="Un sujet ne peut pas être son propre parent")

        if sujet_update.parent_sujet_id:
            parent = db.query(SujetModel).filter(SujetModel.id == sujet_update.parent_sujet_id).first()
            if not parent:
                raise HTTPException(status_code=404, detail="Sujet parent introuvable")

            # Vérifier qu'on ne crée pas de cycle
            current = parent
            while current is not None:
                if current.id == sujet_id:
                    raise HTTPException(status_code=400, detail="Cycle détecté : le parent ne peut pas être un descendant du sujet")
                current = getattr(current, "parent", None)

    # Mettre à jour les champs fournis
    update_data = sujet_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(sujet, field, value)

    db.flush()
    db.refresh(sujet)
    db.commit()

    return sujet_to_dict(sujet)


@app.delete("/api/sujets/{sujet_id}", status_code=200)
def delete_sujet(sujet_id: int, db: Session = Depends(get_db)):
    """Supprimer un sujet (et tous ses enfants et actions en cascade)"""
    sujet = db.query(SujetModel).filter(SujetModel.id == sujet_id).first()
    if not sujet:
        raise HTTPException(status_code=404, detail="Sujet introuvable")

    # Compter les sujets descendants (récursif)
    def count_descendants(s: SujetModel) -> int:
        return 1 + sum(count_descendants(e) for e in s.enfants)

    total_sujets = count_descendants(sujet)
    total_actions = db.query(ActionModel).filter(ActionModel.sujet_id == sujet_id).count()

    titre = sujet.titre
    db.delete(sujet)
    db.flush()
    db.commit()

    return {
        'message': f'Sujet "{titre}" supprimé avec succès',
        'sujet_id': sujet_id,
        'sujets_supprimes': total_sujets,
        'actions_supprimees': total_actions
    }


@app.patch("/api/sujets/{sujet_id}/deplacer", response_model=SujetResponse)
def deplacer_sujet(sujet_id: int, nouveau_parent_id: Optional[int] = None, db: Session = Depends(get_db)):
    """Déplacer un sujet vers un nouveau parent (ou le rendre racine si nouveau_parent_id=null)"""
    sujet = db.query(SujetModel).filter(SujetModel.id == sujet_id).first()
    if not sujet:
        raise HTTPException(status_code=404, detail="Sujet introuvable")

    if nouveau_parent_id == sujet_id:
        raise HTTPException(status_code=400, detail="Un sujet ne peut pas être son propre parent")

    # Vérifier le nouveau parent
    if nouveau_parent_id:
        nouveau_parent = db.query(SujetModel).filter(SujetModel.id == nouveau_parent_id).first()
        if not nouveau_parent:
            raise HTTPException(status_code=404, detail="Nouveau parent introuvable")

        # Vérifier qu'on ne crée pas de cycle
        current = nouveau_parent
        while current is not None:
            if current.id == sujet_id:
                raise HTTPException(status_code=400, detail="Cycle détecté : impossible de déplacer vers un descendant")
            current = getattr(current, "parent", None)

    sujet.parent_sujet_id = nouveau_parent_id
    db.flush()
    db.refresh(sujet)
    db.commit()

    return sujet_to_dict(sujet)


# ========================================
# APIs ACTIONS
# ========================================

@app.post("/api/actions", response_model=ActionResponse, status_code=201)
def create_action(action: ActionCreate, db: Session = Depends(get_db)):
    """Créer une nouvelle action (peut être une sous-action si parent_action_id est fourni)"""
    # Vérifier si le sujet existe
    sujet = db.query(SujetModel).filter(SujetModel.id == action.sujet_id).first()
    if not sujet:
        raise HTTPException(status_code=404, detail="Sujet introuvable")

    # Vérifier si l'action parent existe (si fournie)
    if action.parent_action_id:
        parent_action = db.query(ActionModel).filter(ActionModel.id == action.parent_action_id).first()
        if not parent_action:
            raise HTTPException(status_code=404, detail="Action parente introuvable")

        # Vérifier que l'action parent appartient au même sujet
        if parent_action.sujet_id != action.sujet_id:
            raise HTTPException(status_code=400, detail="L'action parente doit appartenir au même sujet")

    db_action = ActionModel(**action.dict())

    # Calculer depth réel
    db_action.depth = 0
    if db_action.parent_action_id:
        # On attache temporairement le parent pour calculer la profondeur ensuite
        parent = db.query(ActionModel).filter(ActionModel.id == db_action.parent_action_id).first()
        db_action.parent_action = parent
        db_action.depth = action_depth(db_action)

    db.add(db_action)
    db.flush()
    db.refresh(db_action)
    db.commit()

    return action_to_dict(db_action)


@app.get("/api/actions/{action_id}", response_model=ActionResponse)
def get_action(
    action_id: int,
    include_children: bool = Query(False, description="Inclure toutes les sous-actions récursivement"),
    db: Session = Depends(get_db),
):
    """Récupérer une action avec option d'inclure toutes ses sous-actions"""
    action = db.query(ActionModel).filter(ActionModel.id == action_id).first()
    if not action:
        raise HTTPException(status_code=404, detail="Action introuvable")

    return action_to_dict(action, include_children=include_children)


@app.get("/api/sujets/{sujet_id}/actions", response_model=dict)
def get_actions_by_sujet(
    sujet_id: int,
    include_hierarchy: bool = Query(False, description="Inclure toute la hiérarchie des sous-actions"),
    db: Session = Depends(get_db),
):
    """Récupérer toutes les actions racines d'un sujet (sans parent)"""
    sujet = db.query(SujetModel).filter(SujetModel.id == sujet_id).first()
    if not sujet:
        raise HTTPException(status_code=404, detail="Sujet introuvable")

    # Récupérer uniquement les actions de niveau 0 (sans parent)
    actions_racines = db.query(ActionModel).filter(
        ActionModel.sujet_id == sujet_id,
        ActionModel.parent_action_id == None
    ).all()

    return {
        'sujet_id': sujet_id,
        'count': len(actions_racines),
        'actions': [action_to_dict(a, include_children=include_hierarchy) for a in actions_racines]
    }


@app.get("/api/actions/{action_id}/sous-actions", response_model=dict)
def get_sous_actions(action_id: int, db: Session = Depends(get_db)):
    """Récupérer toutes les sous-actions directes d'une action"""
    action = db.query(ActionModel).filter(ActionModel.id == action_id).first()
    if not action:
        raise HTTPException(status_code=404, detail="Action introuvable")

    return {
        'action_id': action_id,
        'titre': action.titre,
        'sous_actions': [action_to_dict(sa) for sa in action.sous_actions]
    }


@app.get("/api/actions/{action_id}/arborescence", response_model=ActionArborescenceResponse)
def get_action_arborescence(action_id: int, db: Session = Depends(get_db)):
    """Récupérer l'arborescence complète (parent et enfants) d'une action"""
    action = db.query(ActionModel).filter(ActionModel.id == action_id).first()
    if not action:
        raise HTTPException(status_code=404, detail="Action introuvable")

    # Remonter jusqu'à l'action racine
    chemin_parents: List[ActionResponse] = []
    current = action
    while getattr(current, "parent_action", None):
        chemin_parents.insert(0, action_to_dict(current.parent_action))
        current = current.parent_action

    return {
        'sujet': sujet_to_dict(action.sujet),
        'chemin_parents': chemin_parents,
        'action_actuelle': action_to_dict(action, include_children=True)
    }


# ========================================
# ENDPOINTS UTILITAIRES
# ========================================

@app.get("/api/health")
def health_check():
    """Vérifier que l'API fonctionne"""
    return {
        'status': 'ok',
        'message': 'API Action Plan opérationnelle',
        'version': '1.0.0'
    }


@app.get("/")
def root():
    """Page d'accueil avec documentation"""
    return {
        'message': "Bienvenue sur l'API Action Plan",
        'documentation': '/docs',
        'endpoints': {
            'sujets': '/api/sujets',
            'actions': '/api/actions',
            'health': '/api/health'
        }
    }


# ========================================
# DÉMARRAGE LOCAL
# ========================================

if __name__ == '__main__':
    import uvicorn
    # IMPORTANT : S'assurer que les tables existent (sinon, utilisez migrations Alembic)
    Base.metadata.create_all(bind=engine)
    uvicorn.run(app, host="0.0.0.0", port=8000)
