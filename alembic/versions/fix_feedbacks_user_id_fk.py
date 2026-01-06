"""fix_feedbacks_user_id_foreignkey

Revision ID: fix_feedbacks_user_id_fk
Revises: ef640c8f7014
Create Date: 2026-01-06 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'fix_feedbacks_user_id_fk'
down_revision: Union[str, Sequence[str], None] = 'ef640c8f7014'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Drop the old ForeignKey constraint (name may need to be adjusted)
    op.drop_constraint('feedbacks_user_id_fkey', 'feedbacks', type_='foreignkey')
    # Create the new ForeignKey constraint referencing "user"
    op.create_foreign_key('feedbacks_user_id_fkey', 'feedbacks', 'user', ['user_id'], ['user_id'])

def downgrade() -> None:
    # Drop the new ForeignKey constraint
    op.drop_constraint('feedbacks_user_id_fkey', 'feedbacks', type_='foreignkey')
    # Re-create the old ForeignKey constraint referencing "users"
    op.create_foreign_key('feedbacks_user_id_fkey', 'feedbacks', 'users', ['user_id'], ['user_id'])
