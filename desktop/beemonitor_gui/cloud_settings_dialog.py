"""
Login Dialog
=============
Login / Register dialog for BeeMonitor cloud account.
"""

import requests
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .cloud_client import (
    DEFAULT_API_URL,
    BeeMonitorCloudClient,
    load_cloud_config,
    save_cloud_config,
)


class LoginDialog(QDialog):
    """Login / Register dialog for BeeMonitor cloud account."""

    logged_in = pyqtSignal(dict)  # emits full config dict on success

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("BeeMonitor Account")
        self.setMinimumWidth(420)
        self.setMinimumHeight(350)

        config = load_cloud_config()

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Tabs: Login / Register
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_login_tab(), "Login")
        self.tabs.addTab(self._create_register_tab(), "Register")
        layout.addWidget(self.tabs)

        # Status message
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Advanced: API URL (collapsed by default)
        advanced_group = QGroupBox("Advanced")
        advanced_group.setCheckable(True)
        advanced_group.setChecked(False)
        adv_layout = QVBoxLayout()
        adv_layout.addWidget(QLabel("API Server:"))
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText(DEFAULT_API_URL)
        self.url_edit.setText(config.get("api_url", DEFAULT_API_URL))
        adv_layout.addWidget(self.url_edit)
        advanced_group.setLayout(adv_layout)
        layout.addWidget(advanced_group)

        layout.addStretch()

        # Cancel button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _create_login_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        layout.addWidget(QLabel("Username or Email:"))
        self.login_username = QLineEdit()
        self.login_username.setPlaceholderText("Enter your username or email")
        layout.addWidget(self.login_username)

        layout.addWidget(QLabel("Password:"))
        self.login_password = QLineEdit()
        self.login_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.login_password.setPlaceholderText("Enter your password")
        layout.addWidget(self.login_password)

        login_btn = QPushButton("Login")
        login_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;"
        )
        login_btn.clicked.connect(self._do_login)
        layout.addWidget(login_btn)

        # Allow Enter key to submit
        self.login_password.returnPressed.connect(login_btn.click)

        layout.addStretch()
        return tab

    def _create_register_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)

        layout.addWidget(QLabel("Username:"))
        self.reg_username = QLineEdit()
        self.reg_username.setPlaceholderText("Choose a username")
        layout.addWidget(self.reg_username)

        layout.addWidget(QLabel("Email:"))
        self.reg_email = QLineEdit()
        self.reg_email.setPlaceholderText("your@email.com")
        layout.addWidget(self.reg_email)

        layout.addWidget(QLabel("Password:"))
        self.reg_password = QLineEdit()
        self.reg_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.reg_password.setPlaceholderText("Min 8 characters")
        layout.addWidget(self.reg_password)

        register_btn = QPushButton("Create Account")
        register_btn.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; padding: 8px;"
        )
        register_btn.clicked.connect(self._do_register)
        layout.addWidget(register_btn)

        self.reg_password.returnPressed.connect(register_btn.click)

        layout.addStretch()
        return tab

    def _get_api_url(self) -> str:
        return self.url_edit.text().strip() or DEFAULT_API_URL

    def _set_status(self, message: str, error: bool = False):
        self.status_label.setText(message)
        color = "#f44336" if error else "#4CAF50"
        self.status_label.setStyleSheet(f"color: {color};")
        QApplication.processEvents()

    def _do_login(self):
        username = self.login_username.text().strip()
        password = self.login_password.text()
        if not username or not password:
            self._set_status("Enter username and password", error=True)
            return

        api_url = self._get_api_url()
        self._set_status("Logging in...")

        try:
            data = BeeMonitorCloudClient.login(api_url, username, password)
            self._on_auth_success(api_url, data)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 401:
                detail = e.response.json().get("detail", "Invalid credentials")
                self._set_status(detail, error=True)
            else:
                self._set_status(f"Login failed: {e}", error=True)
        except requests.ConnectionError:
            self._set_status("Cannot reach server", error=True)
        except Exception as e:
            self._set_status(str(e), error=True)

    def _do_register(self):
        username = self.reg_username.text().strip()
        email = self.reg_email.text().strip()
        password = self.reg_password.text()

        if not username or not email or not password:
            self._set_status("Fill in all fields", error=True)
            return
        if len(password) < 8:
            self._set_status("Password must be at least 8 characters", error=True)
            return

        api_url = self._get_api_url()
        self._set_status("Creating account...")

        try:
            data = BeeMonitorCloudClient.register(api_url, username, email, password)
            self._on_auth_success(api_url, data)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in (400, 409):
                detail = e.response.json().get("detail", "Registration failed")
                self._set_status(detail, error=True)
            else:
                self._set_status(f"Registration failed: {e}", error=True)
        except requests.ConnectionError:
            self._set_status("Cannot reach server", error=True)
        except Exception as e:
            self._set_status(str(e), error=True)

    def _on_auth_success(self, api_url: str, data: dict):
        """Handle successful login or registration."""
        api_key = data["api_key"]
        username = data.get("username", "")
        email = data.get("email", "")
        tier = data.get("tier", "free")

        save_cloud_config(api_url, api_key, username, email, tier)

        config = {
            "api_url": api_url,
            "api_key": api_key,
            "username": username,
            "email": email,
            "tier": tier,
        }
        self.logged_in.emit(config)
        self.accept()
