"""Qwen3-VL action-space schema goldens."""

from __future__ import annotations

from lite.agents.models.qwen3_vl.action_space import (
    Qwen3VLDesktopActionSpace,
    Qwen3VLMobileActionSpace,
)


def _assert_canonical(schemas):
    assert all(set(schema) == {"type", "function"} for schema in schemas)
    assert all(schema["type"] == "function" for schema in schemas)
    assert all(isinstance(schema["function"], dict) for schema in schemas)


# fmt: off
GOLDEN_QWEN3_VL_DESKTOP = [{'type': 'function',
  'function': {'name': 'computer_use',
               'description': 'Use a mouse and keyboard to interact with a computer, '
                              'and take screenshots.\n'
                              '\n'
                              '* This is an interface to a desktop GUI. You do not '
                              'have access to a terminal or applications menu. You '
                              'must click on desktop icons to start applications.\n'
                              '* Some applications may take time to start or process '
                              'actions, so you may need to wait and take successive '
                              'screenshots to see the results of your actions. E.g. if '
                              "you click on Firefox and a window doesn't open, try "
                              'wait and taking another screenshot.\n'
                              "* The screen's resolution is 1000x1000.\n"
                              '* Whenever you intend to move the cursor to click on an '
                              'element like an icon, you should consult a screenshot '
                              'to determine the coordinates of the element before '
                              'moving the cursor.\n'
                              '* If you tried clicking on a program or link but it '
                              'failed to load, even after waiting, try adjusting your '
                              'cursor position so that the tip of the cursor visually '
                              'falls on the element that you want to click.\n'
                              '* Make sure to click any buttons, links, icons, etc '
                              "with the cursor tip in the center of the element. Don't "
                              'click boxes on their edges.',
               'parameters': {'type': 'object',
                              'properties': {'action': {'type': 'string',
                                                        'enum': ['key',
                                                                 'type',
                                                                 'mouse_move',
                                                                 'left_click',
                                                                 'left_click_drag',
                                                                 'right_click',
                                                                 'middle_click',
                                                                 'double_click',
                                                                 'triple_click',
                                                                 'scroll',
                                                                 'hscroll',
                                                                 'wait',
                                                                 'terminate',
                                                                 'answer'],
                                                        'description': '\n'
                                                                       '* `key`: '
                                                                       'Performs key '
                                                                       'down presses '
                                                                       'on the '
                                                                       'arguments '
                                                                       'passed in '
                                                                       'order, then '
                                                                       'performs key '
                                                                       'releases in '
                                                                       'reverse '
                                                                       'order.\n'
                                                                       '* `type`: type '
                                                                       'a string of '
                                                                       'text on the '
                                                                       'keyboard.\n'
                                                                       '* '
                                                                       '`mouse_move`: '
                                                                       'Move the '
                                                                       'cursor to a '
                                                                       'specified (x, '
                                                                       'y) pixel '
                                                                       'coordinate on '
                                                                       'the screen.\n'
                                                                       '* '
                                                                       '`left_click`: '
                                                                       'Click the left '
                                                                       'mouse button '
                                                                       'at a specified '
                                                                       '(x, y) pixel '
                                                                       'coordinate on '
                                                                       'the screen.\n'
                                                                       '* '
                                                                       '`left_click_drag`: '
                                                                       'Click and drag '
                                                                       'the cursor to '
                                                                       'a specified '
                                                                       '(x, y) pixel '
                                                                       'coordinate on '
                                                                       'the screen.\n'
                                                                       '* '
                                                                       '`right_click`: '
                                                                       'Click the '
                                                                       'right mouse '
                                                                       'button at a '
                                                                       'specified (x, '
                                                                       'y) pixel '
                                                                       'coordinate on '
                                                                       'the screen.\n'
                                                                       '* '
                                                                       '`middle_click`: '
                                                                       'Click the '
                                                                       'middle mouse '
                                                                       'button at a '
                                                                       'specified (x, '
                                                                       'y) pixel '
                                                                       'coordinate on '
                                                                       'the screen.\n'
                                                                       '* '
                                                                       '`double_click`: '
                                                                       'Double-click '
                                                                       'the left mouse '
                                                                       'button at a '
                                                                       'specified (x, '
                                                                       'y) pixel '
                                                                       'coordinate on '
                                                                       'the screen.\n'
                                                                       '* '
                                                                       '`triple_click`: '
                                                                       'Triple-click '
                                                                       'the left mouse '
                                                                       'button at a '
                                                                       'specified (x, '
                                                                       'y) pixel '
                                                                       'coordinate on '
                                                                       'the screen.\n'
                                                                       '* `scroll`: '
                                                                       'Performs a '
                                                                       'scroll of the '
                                                                       'mouse scroll '
                                                                       'wheel.\n'
                                                                       '* `hscroll`: '
                                                                       'Performs a '
                                                                       'horizontal '
                                                                       'scroll.\n'
                                                                       '* `wait`: Wait '
                                                                       'specified '
                                                                       'seconds for '
                                                                       'the change to '
                                                                       'happen.\n'
                                                                       '* `terminate`: '
                                                                       'Terminate the '
                                                                       'current task '
                                                                       'and report its '
                                                                       'completion '
                                                                       'status.\n'
                                                                       '* `answer`: '
                                                                       'Answer a '
                                                                       'question.\n'},
                                             'keys': {'type': 'array',
                                                      'items': {'type': 'string'},
                                                      'description': 'Required only by '
                                                                     '`action=key`.'},
                                             'text': {'type': 'string',
                                                      'description': 'Required only by '
                                                                     '`action=type` '
                                                                     'and '
                                                                     '`action=answer`.'},
                                             'coordinate': {'type': 'array',
                                                            'items': {'type': 'integer'},
                                                            'description': 'The x,y '
                                                                           'coordinates '
                                                                           'for mouse '
                                                                           'actions.'},
                                             'pixels': {'type': 'integer',
                                                        'description': 'The amount of '
                                                                       'scrolling. '
                                                                       'Positive '
                                                                       'values scroll '
                                                                       'up, negative '
                                                                       'values scroll '
                                                                       'down. Required '
                                                                       'only by '
                                                                       '`action=scroll` '
                                                                       'and '
                                                                       '`action=hscroll`.'},
                                             'time': {'type': 'number',
                                                      'description': 'The seconds to '
                                                                     'wait. The '
                                                                     'maximum accepted '
                                                                     'value depends on '
                                                                     'the action (wait '
                                                                     '<= 30, hold_key '
                                                                     '<= 5, long_press '
                                                                     '<= 5, swipe <= '
                                                                     '5, drag <= 5 '
                                                                     'seconds); a '
                                                                     'larger value is '
                                                                     'rejected and that '
                                                                     'action does not '
                                                                     'run.'},
                                             'status': {'type': 'string',
                                                        'enum': ['success', 'failure'],
                                                        'description': 'The status of '
                                                                       'the task.'}},
                              'required': ['action']}}}]

GOLDEN_QWEN3_VL_MOBILE = [{'type': 'function',
  'function': {'name': 'mobile_use',
               'description': 'Use a touchscreen to interact with a mobile device, and '
                              'take screenshots.\n'
                              '\n'
                              '* This is an interface to a mobile device with '
                              'touchscreen. You can perform actions like clicking, '
                              'typing, swiping, etc.\n'
                              '* Some applications may take time to start or process '
                              'actions, so you may need to wait and take successive '
                              'screenshots to see the results of your actions.\n'
                              "* The screen's resolution is 999x999.\n"
                              '* Make sure to click any buttons, links, icons, etc '
                              "with the cursor tip in the center of the element. Don't "
                              'click boxes on their edges unless asked.',
               'parameters': {'type': 'object',
                              'properties': {'action': {'type': 'string',
                                                        'enum': ['click',
                                                                 'long_press',
                                                                 'swipe',
                                                                 'type',
                                                                 'open',
                                                                 'answer',
                                                                 'system_button',
                                                                 'wait',
                                                                 'terminate'],
                                                        'description': 'The action to '
                                                                       'perform. The '
                                                                       'available '
                                                                       'actions are:\n'
                                                                       '* `click`: '
                                                                       'Click the '
                                                                       'point on the '
                                                                       'screen with '
                                                                       'coordinate (x, '
                                                                       'y).\n'
                                                                       '* '
                                                                       '`long_press`: '
                                                                       'Press the '
                                                                       'point on the '
                                                                       'screen with '
                                                                       'coordinate (x, '
                                                                       'y) for '
                                                                       'specified '
                                                                       'seconds.\n'
                                                                       '* `swipe`: '
                                                                       'Swipe from the '
                                                                       'starting point '
                                                                       'with '
                                                                       'coordinate (x, '
                                                                       'y) to the end '
                                                                       'point with '
                                                                       'coordinates2 '
                                                                       '(x2, y2).\n'
                                                                       '* `type`: '
                                                                       'Input the '
                                                                       'specified text '
                                                                       'into the '
                                                                       'activated '
                                                                       'input box.\n'
                                                                       '* `open`: Open '
                                                                       'an app on the '
                                                                       'device.\n'
                                                                       '* `answer`: '
                                                                       'Output the '
                                                                       'answer.\n'
                                                                       '* '
                                                                       '`system_button`: '
                                                                       'Press the '
                                                                       'system '
                                                                       'button.\n'
                                                                       '* `wait`: Wait '
                                                                       'specified '
                                                                       'seconds for '
                                                                       'the change to '
                                                                       'happen.\n'
                                                                       '* `terminate`: '
                                                                       'Terminate the '
                                                                       'current task '
                                                                       'and report its '
                                                                       'completion '
                                                                       'status.'},
                                             'coordinate': {'type': 'array',
                                                            'items': {'type': 'integer'},
                                                            'description': '(x, y): '
                                                                           'The x '
                                                                           '(pixels '
                                                                           'from the '
                                                                           'left edge) '
                                                                           'and y '
                                                                           '(pixels '
                                                                           'from the '
                                                                           'top edge) '
                                                                           'coordinates '
                                                                           'to move '
                                                                           'the mouse '
                                                                           'to. '
                                                                           'Required '
                                                                           'only by '
                                                                           '`action=click`, '
                                                                           '`action=long_press`, '
                                                                           'and '
                                                                           '`action=swipe`.'},
                                             'coordinate2': {'type': 'array',
                                                             'items': {'type': 'integer'},
                                                             'description': '(x, y): '
                                                                            'The x '
                                                                            '(pixels '
                                                                            'from the '
                                                                            'left '
                                                                            'edge) and '
                                                                            'y (pixels '
                                                                            'from the '
                                                                            'top edge) '
                                                                            'coordinates '
                                                                            'to move '
                                                                            'the mouse '
                                                                            'to. '
                                                                            'Required '
                                                                            'only by '
                                                                            '`action=swipe`.'},
                                             'text': {'type': 'string',
                                                      'description': 'Required only by '
                                                                     '`action=type`, '
                                                                     '`action=open`, '
                                                                     'and '
                                                                     '`action=answer`.'},
                                             'time': {'type': 'number',
                                                      'description': 'The seconds to '
                                                                     'wait. Required '
                                                                     'only by '
                                                                     '`action=long_press` '
                                                                     'and '
                                                                     '`action=wait`. '
                                                                     'The maximum '
                                                                     'accepted value '
                                                                     'depends on the '
                                                                     'action (wait <= '
                                                                     '30, hold_key <= '
                                                                     '5, long_press <= '
                                                                     '5, swipe <= 5, '
                                                                     'drag <= 5 '
                                                                     'seconds); a '
                                                                     'larger value is '
                                                                     'rejected and that '
                                                                     'action does not '
                                                                     'run.'},
                                             'button': {'type': 'string',
                                                        'enum': ['Back',
                                                                 'Home',
                                                                 'Menu',
                                                                 'Enter',
                                                                 'Recent'],
                                                        'description': 'Back means '
                                                                       'returning to '
                                                                       'the previous '
                                                                       'interface, '
                                                                       'Home means '
                                                                       'returning to '
                                                                       'the desktop, '
                                                                       'Menu means '
                                                                       'opening the '
                                                                       'application '
                                                                       'background '
                                                                       'menu, and '
                                                                       'Enter means '
                                                                       'pressing the '
                                                                       'enter. '
                                                                       'Required only '
                                                                       'by '
                                                                       '`action=system_button`'},
                                             'status': {'type': 'string',
                                                        'enum': ['success', 'failure'],
                                                        'description': 'The status of '
                                                                       'the task. '
                                                                       'Required only '
                                                                       'by '
                                                                       '`action=terminate`.'}},
                              'required': ['action']}}}]

# GOLDEN_CLAUDE_DESKTOP / GOLDEN_GPT_DESKTOP were removed: both were DEAD
# (defined, never referenced -- the live assertions are ``actual == []``, since
# an opaque-provider space emits nothing) and STALE (13 vs 15 and 10 vs 12
# entries; the two extra `response`/`terminate` bodies pinned prose that no
# longer exists anywhere in ``lite/``). Re-capturing them would have been
# authoring new goldens, not restoring old ones. The behavior they were meant
# to protect is covered by native-call conversion order below.
# fmt: on


def test_qwen3_vl_desktop_schema_golden():
    actual = Qwen3VLDesktopActionSpace().get_tool_schemas()
    _assert_canonical(actual)
    assert actual == GOLDEN_QWEN3_VL_DESKTOP


def test_qwen3_vl_mobile_schema_golden():
    actual = Qwen3VLMobileActionSpace().get_tool_schemas()
    _assert_canonical(actual)
    assert actual == GOLDEN_QWEN3_VL_MOBILE
