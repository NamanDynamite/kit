"""remove_validator_type_and_points_awarded_from_validations

Revision ID: ef640c8f7014
Revises: 1b17080404b4
Create Date: 2026-01-02 11:21:31.581693

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ef640c8f7014'
down_revision: Union[str, Sequence[str], None] = '1b17080404b4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Drop columns from validations table only if they exist
    from sqlalchemy import inspect
    bind = op.get_bind()
    inspector = inspect(bind)
    validation_columns = [col['name'] for col in inspector.get_columns('validations')]
    if 'validator_type' in validation_columns:
        op.drop_column('validations', 'validator_type')
    if 'points_awarded' in validation_columns:
        op.drop_column('validations', 'points_awarded')


def downgrade() -> None:
    """Downgrade schema."""
    # Re-add columns to validations table
    op.add_column('validations', sa.Column('validator_type', sa.String(), nullable=True))
    op.add_column('validations', sa.Column('points_awarded', sa.Integer(), nullable=True))
