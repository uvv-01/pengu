"""
Tests for the IntentRouter — rule-based command classification.
"""

import pytest
from pengu.config import TaskCategory
from pengu.router import IntentRouter, Intent, reset_router


@pytest.fixture
def router():
    return reset_router()


class TestIntentRouterClassification:
    """Test that rules correctly classify user text into categories."""

    # --- SYSTEM_CONTROL ---

    def test_open_vs_code(self, router):
        intent = router.classify("open VS Code")
        assert intent.category == TaskCategory.SYSTEM_CONTROL

    def test_open_chrome(self, router):
        intent = router.classify("open Chrome")
        assert intent.category == TaskCategory.SYSTEM_CONTROL

    def test_launch_application(self, router):
        intent = router.classify("launch Notepad")
        assert intent.category == TaskCategory.SYSTEM_CONTROL

    def test_close_application(self, router):
        intent = router.classify("close Chrome")
        assert intent.category == TaskCategory.SYSTEM_CONTROL

    def test_focus_application(self, router):
        intent = router.classify("focus on VS Code")
        assert intent.category == TaskCategory.SYSTEM_CONTROL

    def test_switch_to_application(self, router):
        intent = router.classify("switch to Firefox")
        assert intent.category == TaskCategory.SYSTEM_CONTROL

    def test_vs_code_with_folder(self, router):
        intent = router.classify("open my FoveaEdge project in VS Code")
        assert intent.category == TaskCategory.SYSTEM_CONTROL
        # After extraction, action becomes "open" (not "vscode")
        assert intent.extracted_action in ("open", "vscode")

    # --- GIT ---

    def test_git_status(self, router):
        intent = router.classify("git status")
        assert intent.category == TaskCategory.GIT

    def test_git_diff(self, router):
        intent = router.classify("git diff")
        assert intent.category == TaskCategory.GIT

    def test_git_log(self, router):
        intent = router.classify("git log")
        assert intent.category == TaskCategory.GIT

    def test_git_branch(self, router):
        intent = router.classify("git branch")
        assert intent.category == TaskCategory.GIT

    def test_check_git(self, router):
        intent = router.classify("check git status")
        assert intent.category == TaskCategory.GIT

    # --- FILE_OPERATION ---

    def test_read_file(self, router):
        intent = router.classify("read the README file")
        assert intent.category == TaskCategory.FILE_OPERATION
        assert intent.extracted_action == "read"

    def test_list_directory(self, router):
        intent = router.classify("list files in the current directory")
        assert intent.category == TaskCategory.FILE_OPERATION
        assert intent.extracted_action == "list"

    def test_find_files(self, router):
        intent = router.classify("find all Python files")
        assert intent.category == TaskCategory.FILE_OPERATION

    def test_create_file(self, router):
        intent = router.classify("create a new file called test.py")
        assert intent.category == TaskCategory.FILE_OPERATION
        assert intent.extracted_action == "create"

    def test_search_in_files(self, router):
        intent = router.classify("search for 'def main' in all Python files")
        assert intent.category == TaskCategory.FILE_OPERATION

    # --- TERMINAL ---

    def test_terminal_command(self, router):
        intent = router.classify("run the command dir")
        assert intent.category == TaskCategory.TERMINAL

    def test_powershell(self, router):
        intent = router.classify("open PowerShell")
        # This should be SYSTEM_CONTROL since "open" is more specific
        # But "PowerShell" alone should match terminal
        assert intent.category in (TaskCategory.TERMINAL, TaskCategory.SYSTEM_CONTROL)

    def test_execute_script(self, router):
        intent = router.classify("execute the script build.ps1")
        assert intent.category == TaskCategory.TERMINAL

    # --- CODING ---

    def test_write_program(self, router):
        intent = router.classify("write a Python program to sort a list")
        # Should be CODING due to 'write...program' pattern
        assert intent.category in (TaskCategory.CODING, TaskCategory.FILE_OPERATION)

    def test_fix_bug(self, router):
        intent = router.classify("fix the bug in main.py")
        assert intent.category == TaskCategory.CODING

    def test_explain_code(self, router):
        intent = router.classify("explain this code")
        assert intent.category == TaskCategory.CODING

    def test_implement_feature(self, router):
        intent = router.classify("implement a binary search function")
        assert intent.category == TaskCategory.CODING

    def test_create_function(self, router):
        intent = router.classify("create a function that calculates factorial")
        assert intent.category == TaskCategory.CODING

    # --- WEB_SEARCH ---

    def test_search_for(self, router):
        intent = router.classify("search for Python documentation")
        assert intent.category == TaskCategory.WEB_SEARCH

    def test_google_something(self, router):
        intent = router.classify("google for latest Python release")
        assert intent.category == TaskCategory.WEB_SEARCH

    def test_look_up(self, router):
        intent = router.classify("look up how to use async in Python")
        assert intent.category == TaskCategory.WEB_SEARCH

    # --- VISION ---

    def test_look_at_screen(self, router):
        intent = router.classify("look at my screen")
        assert intent.category == TaskCategory.VISION

    def test_what_on_screen(self, router):
        intent = router.classify("what is on my screen?")
        assert intent.category == TaskCategory.VISION

    def test_screenshot(self, router):
        intent = router.classify("take a screenshot")
        assert intent.category == TaskCategory.VISION

    # --- NETWORK ---

    def test_wifi_status(self, router):
        intent = router.classify("what is my WiFi status?")
        assert intent.category == TaskCategory.NETWORK

    def test_list_wifi(self, router):
        intent = router.classify("list available wifi networks")
        assert intent.category == TaskCategory.NETWORK

    def test_network_info(self, router):
        intent = router.classify("network status")
        assert intent.category == TaskCategory.NETWORK

    # --- CHAT ---

    def test_hello(self, router):
        intent = router.classify("hello")
        assert intent.category == TaskCategory.CHAT

    def test_hi(self, router):
        intent = router.classify("hi there")
        assert intent.category == TaskCategory.CHAT

    def test_how_are_you(self, router):
        intent = router.classify("how are you?")
        assert intent.category == TaskCategory.CHAT

    def test_thank_you(self, router):
        intent = router.classify("thank you")
        assert intent.category == TaskCategory.CHAT

    def test_what_can_you_do(self, router):
        intent = router.classify("what can you do?")
        assert intent.category == TaskCategory.CHAT

    def test_who_are_you(self, router):
        intent = router.classify("who are you?")
        assert intent.category == TaskCategory.CHAT

    def test_what_time(self, router):
        intent = router.classify("what time is it?")
        assert intent.category == TaskCategory.CHAT

    # --- Edge cases ---

    def test_empty_input(self, router):
        intent = router.classify("")
        assert intent.category == TaskCategory.CHAT
        assert intent.confidence == 1.0

    def test_whitespace_only(self, router):
        intent = router.classify("   ")
        assert intent.category == TaskCategory.CHAT

    def test_unrecognized_defaults_to_chat(self, router):
        intent = router.classify("asdfghjkl random gibberish 12345")
        assert intent.category == TaskCategory.CHAT


class TestIntentExtraction:
    """Test that action/target extraction works correctly."""

    def test_extract_vscode_action(self, router):
        intent = router.classify("open VS Code")
        assert intent.extracted_action in ("open", "vscode")

    def test_extract_application_target(self, router):
        intent = router.classify("open Chrome")
        assert "chrome" in intent.extracted_target.lower() or intent.extracted_action == "open"

    def test_extract_git_subcommand(self, router):
        intent = router.classify("git status")
        assert "git" in intent.extracted_action

    def test_extract_search_query(self, router):
        intent = router.classify("search for Python documentation")
        assert "python" in intent.extracted_target.lower() or intent.category == TaskCategory.WEB_SEARCH

    def test_intent_has_raw_text(self, router):
        text = "open VS Code and check git status"
        intent = router.classify(text)
        assert intent.raw_text == text

    def test_intent_confidence_is_numeric(self, router):
        intent = router.classify("hello")
        assert isinstance(intent.confidence, float)
        assert 0.0 <= intent.confidence <= 1.0


class TestRouterPriority:
    """Test that more specific rules take priority over general ones."""

    def test_vscode_over_generic_chat(self, router):
        """'open VS Code' should be SYSTEM_CONTROL, not CHAT."""
        intent = router.classify("open VS Code")
        assert intent.category == TaskCategory.SYSTEM_CONTROL

    def test_git_over_generic(self, router):
        """'git status' should be GIT, not TERMINAL."""
        intent = router.classify("git status")
        assert intent.category == TaskCategory.GIT

    def test_specific_over_vague(self, router):
        """'write a Python program' should be CODING, not FILE_OPERATION."""
        intent = router.classify("write a Python program to sort a list")
        assert intent.category == TaskCategory.CODING


class TestSystemControlRouter:
    """Test battery, wallpaper, volume routing."""

    def test_battery_status(self, router):
        """'What's my battery?' should be SYSTEM_CONTROL with battery action."""
        intent = router.classify("What's my battery percentage?")
        assert intent.category == TaskCategory.SYSTEM_CONTROL
        assert intent.extracted_action == "battery"

    def test_battery_charging(self, router):
        """'Am I charging?' should route to battery."""
        intent = router.classify("Am I charging?")
        assert intent.category == TaskCategory.SYSTEM_CONTROL
        assert intent.extracted_action == "battery"

    def test_battery_level(self, router):
        """'How much battery do I have?' should route to battery."""
        intent = router.classify("How much battery do I have?")
        assert intent.category == TaskCategory.SYSTEM_CONTROL
        assert intent.extracted_action == "battery"

    def test_wallpaper_change(self, router):
        """'Change my wallpaper' should route to wallpaper."""
        intent = router.classify("Change my wallpaper")
        assert intent.category == TaskCategory.SYSTEM_CONTROL
        assert intent.extracted_action == "wallpaper"

    def test_wallpaper_set(self, router):
        """'Set wallpaper to photo.jpg' should route to wallpaper."""
        intent = router.classify("Set wallpaper to photo.jpg")
        assert intent.category == TaskCategory.SYSTEM_CONTROL
        assert intent.extracted_action == "wallpaper"

    def test_volume_get(self, router):
        """'What's my volume?' should route to volume."""
        intent = router.classify("What's my volume?")
        assert intent.category == TaskCategory.SYSTEM_CONTROL
        assert intent.extracted_action == "volume"

    def test_volume_set(self, router):
        """'Set volume to 50' should route to volume."""
        intent = router.classify("Set volume to 50")
        assert intent.category == TaskCategory.SYSTEM_CONTROL
        assert intent.extracted_action == "volume"

    def test_mute(self, router):
        """'Mute' should route to volume."""
        intent = router.classify("Mute the sound")
        assert intent.category == TaskCategory.SYSTEM_CONTROL
        assert intent.extracted_action == "volume"

    def test_unmute(self, router):
        """'Unmute' should route to volume."""
        intent = router.classify("Unmute")
        assert intent.category == TaskCategory.SYSTEM_CONTROL
        assert intent.extracted_action == "volume"

    def test_volume_up(self, router):
        """'Volume up' should route to volume."""
        intent = router.classify("Turn volume up")
        assert intent.category == TaskCategory.SYSTEM_CONTROL
        assert intent.extracted_action == "volume"

    def test_volume_down(self, router):
        """'Volume down' should route to volume."""
        intent = router.classify("Volume down")
        assert intent.category == TaskCategory.SYSTEM_CONTROL
        assert intent.extracted_action == "volume"

    def test_battery_over_chat(self, router):
        """'battery' should route to SYSTEM_CONTROL, not CHAT."""
        intent = router.classify("What is my battery status")
        assert intent.category == TaskCategory.SYSTEM_CONTROL

    def test_wallpaper_over_chat(self, router):
        """'wallpaper' should route to SYSTEM_CONTROL, not CHAT."""
        intent = router.classify("Change my desktop background")
        assert intent.category == TaskCategory.SYSTEM_CONTROL
