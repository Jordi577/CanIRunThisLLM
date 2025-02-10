from django import template

register = template.Library()

@register.filter
def index(sequence, i):
    try:
        return sequence[int(i)]
    except (IndexError, ValueError, TypeError):
        return ''
