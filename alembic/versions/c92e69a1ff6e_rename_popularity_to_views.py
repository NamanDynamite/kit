"""rename_popularity_to_views

Revision ID: c92e69a1ff6e
Revises: 5b43c406264f
Create Date: 2026-01-27 12:13:09.533172

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c92e69a1ff6e'
down_revision: Union[str, Sequence[str], None] = '5b43c406264f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column('recipes', 'popularity', new_column_name='views')


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column('recipes', 'views', new_column_name='popularity')
