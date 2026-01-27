"""add_popularity_column_to_recipes

Revision ID: 5b43c406264f
Revises: f552b8fc63d4
Create Date: 2026-01-27 12:00:53.475407

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5b43c406264f'
down_revision: Union[str, Sequence[str], None] = 'f552b8fc63d4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('recipes', sa.Column('popularity', sa.Integer(), nullable=True, server_default='0'))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('recipes', 'popularity')
