from Module.schemas.event import EventPlanRequest, EventPlanResponse

class EventService:
    """Service for event-related business logic."""
    
    def plan_event(self, request: EventPlanRequest) -> EventPlanResponse:
        """Plan an event and return details."""
        return EventPlanResponse(
            event=request.event_name,
            user_id=request.user_id,
            guest_count=request.guest_count,
            budget_per_person=request.budget_per_person,
            dietary=request.dietary
        )
