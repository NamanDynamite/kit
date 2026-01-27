"""move_views_from_recipes_to_recipe_versions

Revision ID: 33d9e1d02e3f
Revises: c92e69a1ff6e
Create Date: 2026-01-27 12:28:32.513567

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '33d9e1d02e3f'
down_revision: Union[str, Sequence[str], None] = 'c92e69a1ff6e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add views column to recipe_versions table
    op.add_column('recipe_versions', sa.Column('views', sa.Integer(), server_default='0', nullable=False))
    
    # Drop views column from recipes table (if it exists)
    # Using batch mode for compatibility
    with op.batch_alter_table('recipes') as batch_op:
        batch_op.drop_column('views')


def downgrade() -> None:
    """Downgrade schema."""
    # Add views column back to recipes table
    op.add_column('recipes', sa.Column('views', sa.Integer(), server_default='0', nullable=False))
    
    # Drop views column from recipe_versions table
    with op.batch_alter_table('recipe_versions') as batch_op:
        batch_op.drop_column('views')
