from typing import List
from sqlalchemy.orm import Session

from Module.database import Role
from Module.schemas.role import RoleCreate, RoleResponse

class RoleService:
    """Service for role-related business logic."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_role(self, role_id: str) -> RoleResponse:
        """Get role by ID."""
        db_role = self.db.query(Role).filter(Role.role_id == role_id).first()
        if not db_role:
            raise ValueError(f"Role '{role_id}' not found")
        
        return RoleResponse(
            role_id=db_role.role_id,
            role_name=db_role.role_name,
            description=db_role.description
        )
    
    def create_role(self, role: RoleCreate) -> RoleResponse:
        """Create a new role."""
        existing = self.db.query(Role).filter(Role.role_id == role.role_id).first()
        if existing:
            raise ValueError(f"Role with role_id '{role.role_id}' already exists")
        
        db_role = Role(
            role_id=role.role_id,
            role_name=role.role_name,
            description=role.description
        )
        self.db.add(db_role)
        self.db.commit()
        self.db.refresh(db_role)
        
        return RoleResponse(
            role_id=db_role.role_id,
            role_name=db_role.role_name,
            description=db_role.description
        )
    
    def list_roles(self) -> List[RoleResponse]:
        """List all roles."""
        roles = self.db.query(Role).all()
        return [RoleResponse(
            role_id=r.role_id, 
            role_name=r.role_name, 
            description=r.description
        ) for r in roles]
