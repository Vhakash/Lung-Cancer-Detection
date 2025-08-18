"""
Session state management utilities.
"""
import streamlit as st
from typing import Any, Dict, Optional
from config import MAX_HISTORY_ENTRIES
from logger import logger


class SessionManager:
    """Manages Streamlit session state with type safety and validation."""

    @staticmethod
    def initialize_session_state():
        """Initialize all required session state variables."""
        defaults = {
            'analysis_history': [],
            'show_history': False,
            'show_model_comparison': False,
            'show_patient_list': False,
            'show_patient_form': False,
            'selected_patient_id': None,
            'model': None,
            'model_option': 'Basic CNN',
            'current_page': 'analyze'
        }

        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
                logger.debug(
                    f"Initialized session state: {key} = {default_value}")

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """
        Safely get a value from session state.

        Args:
            key: Session state key
            default: Default value if key doesn't exist

        Returns:
            Value from session state or default
        """
        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any) -> None:
        """
        Set a value in session state.

        Args:
            key: Session state key
            value: Value to set
        """
        st.session_state[key] = value
        logger.debug(f"Set session state: {key} = {type(value).__name__}")

    @staticmethod
    def clear_navigation_flags():
        """Clear all navigation-related flags."""
        navigation_flags = [
            'show_history',
            'show_model_comparison',
            'show_patient_list',
            'show_patient_form'
        ]

        for flag in navigation_flags:
            st.session_state[flag] = False

    @staticmethod
    def add_to_analysis_history(analysis_data: Dict[str, Any]) -> None:
        """
        Add analysis to history with size limit.

        Args:
            analysis_data: Dictionary containing analysis information
        """
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []

        st.session_state.analysis_history.append(analysis_data)

        # Keep only the most recent entries
        if len(st.session_state.analysis_history) > MAX_HISTORY_ENTRIES:
            st.session_state.analysis_history = st.session_state.analysis_history[-MAX_HISTORY_ENTRIES:]

        logger.info(
            f"Added analysis to history. Total entries: {len(st.session_state.analysis_history)}")

    @staticmethod
    def clear_analysis_history() -> None:
        """Clear the analysis history."""
        st.session_state.analysis_history = []
        logger.info("Cleared analysis history")

    @staticmethod
    def get_analysis_history() -> list:
        """
        Get the analysis history.

        Returns:
            List of analysis history entries
        """
        return st.session_state.get('analysis_history', [])

    @staticmethod
    def navigate_to(page: str) -> None:
        """
        Navigate to a specific page by setting appropriate flags.

        Args:
            page: Page to navigate to ('analyze', 'patients', 'history', 'compare')
        """
        SessionManager.clear_navigation_flags()

        if page == 'patients':
            st.session_state.show_patient_list = True
        elif page == 'history':
            st.session_state.show_history = True
        elif page == 'compare':
            st.session_state.show_model_comparison = True
        # 'analyze' is the default state when all flags are False

        st.session_state.current_page = page
        logger.debug(f"Navigated to page: {page}")

    @staticmethod
    def reset_patient_selection():
        """Reset patient-related session state."""
        st.session_state.selected_patient_id = None
        st.session_state.show_patient_form = False
        logger.debug("Reset patient selection")


# Create global session manager instance
session_manager = SessionManager()
