# Router modules for KitchenMind API
# Import order matters - routers register endpoints on the shared api_router

from . import base
from . import auth
from . import public
from . import users
from . import roles
from . import admin
from . import recipes
from . import events

__all__ = ['auth', 'public', 'users', 'roles', 'admin', 'recipes', 'events']
