"""Adapt Streamlit 1.50 AppTest to single-select segmented controls.

Its ButtonGroup test wrapper expects a list, while the actual single-select
widget stores a scalar string in session state. This changes tests only.
"""

from streamlit.testing.v1.element_tree import ButtonGroup


if not getattr(ButtonGroup, '_single_select_compat', False):
    _original_value = ButtonGroup.value.fget

    def _list_value(widget):
        value = _original_value(widget)
        return [value] if isinstance(value, str) else value

    ButtonGroup.value = property(_list_value)
    ButtonGroup._single_select_compat = True
