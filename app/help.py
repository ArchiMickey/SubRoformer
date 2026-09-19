from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)


class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SubStemStudio Help")
        self.setObjectName("helpDialog")
        self.resize(720, 790)
        self.setStyleSheet("""
            QDialog#helpDialog { background: #24273a; }
            QLabel { background: transparent; font-size: 13px; }
            QLabel#helpTitle { font-size: 28px; font-weight: 600; }
            QLabel#helpSubtitle { color: #a5adcb; }
            QLabel#helpSection { color: #c6a0f6; font-size: 14px; font-weight: 600; }
            QLabel#helpKey { background: #363a4f; border: 1px solid #494d64;
                border-radius: 6px; padding: 6px 10px; color: #b7bdf8; font-weight: 600; }
            QLabel#helpBadge { background: #283b3c; color: #8bd5ca;
                border-radius: 6px; padding: 8px 12px; font-weight: 600; }
            QLabel#helpError { background: #302b39; color: #eed49f;
                border-radius: 6px; padding: 10px; font-family: monospace; font-size: 11px; }
            QFrame#helpCard { background: #1e2030; border: 1px solid #363a4f; border-radius: 12px; }
            QScrollArea { border: none; background: transparent; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(8)
        title = QLabel("A little help, a better session.")
        title.setObjectName("helpTitle")
        title.setWordWrap(True)
        layout.addWidget(title)
        subtitle = QLabel("SUBSTEMSTUDIO  /  QUICK REFERENCE")
        subtitle.setObjectName("helpSubtitle")
        layout.addWidget(subtitle)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        sections = QVBoxLayout(content)
        sections.setContentsMargins(0, 12, 8, 12)
        sections.setSpacing(14)
        scroll.setWidget(content)
        layout.addWidget(scroll)

        shortcuts = self.add_card(sections, "01  /  Keyboard & playback")
        keys = QGridLayout()
        keys.setHorizontalSpacing(18)
        keys.setVerticalSpacing(8)
        keys.setColumnStretch(1, 1)
        for row, (key, action) in enumerate((
            ("Space", "Play / pause"),
            ("1", "Select Mixture — all stems"),
            ("2 / 3 / 4 / 5", "Toggle Vocals / Bass / Drums / Others"),
            ("Shift+2–5", "Solo a stem · Shift-click works too"),
            ("`  backtick", "Mute all without pausing"),
            ("~  tilde", "Invert enabled stems"),
        )):
            keycap = QLabel(key)
            keycap.setObjectName("helpKey")
            keycap.setAlignment(Qt.AlignmentFlag.AlignCenter)
            keys.addWidget(keycap, row, 0)
            description = QLabel(action)
            description.setWordWrap(True)
            keys.addWidget(description, row, 1)
        shortcuts.addLayout(keys)
        self.add_text(shortcuts, "Click a spectrogram to seek and play. Drag to pan; scroll to zoom. "
                      "Text and numeric fields keep normal keyboard input.")

        requirements = self.add_card(sections, "02  /  Mac requirements")
        badge = QLabel("macOS 15+  ·  Apple Silicon")
        badge.setObjectName("helpBadge")
        requirements.addWidget(badge)
        self.add_text(requirements, "This packaged app requires macOS 15 (Sequoia) or later, "
                      "even in CPU mode. Core ML also requires macOS 15+.")

        troubleshooting = self.add_card(sections, "03  /  MPS unavailable?")
        error = self.add_text(troubleshooting,
            "ValueError: MPS is unavailable; use an MPS-enabled PyTorch on a supported Mac or --device cpu")
        error.setObjectName("helpError")
        error.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.add_text(troubleshooting, "<b>1. Check macOS.</b> Open Apple menu → About This Mac "
                      "and install available macOS updates.")
        self.add_text(troubleshooting, "<b>2. Check support.</b> MPS needs supported GPU hardware "
                      "and an MPS-enabled PyTorch build. Updating macOS alone may not resolve it.")
        self.add_text(troubleshooting, "<b>3. Use CPU instead.</b> Select <b>cpu</b> in the app's "
                      "Device field, or use <b>--device cpu</b> in the CLI. "
                      "The packaged app still requires macOS 15+.")
        sections.addStretch()
        footer = QHBoxLayout()
        hint = QLabel("Esc to close · Back to making music")
        hint.setObjectName("helpSubtitle")
        footer.addWidget(hint)
        footer.addStretch()
        close = QPushButton("Got it")
        close.setObjectName("primary")
        close.setDefault(True)
        close.clicked.connect(self.accept)
        footer.addWidget(close)
        layout.addLayout(footer)
        available = self.screen().availableGeometry()
        self.resize(min(720, available.width() - 40), min(900, available.height() - 80))

    def add_card(self, layout, title):
        card = QFrame()
        card.setObjectName("helpCard")
        contents = QVBoxLayout(card)
        contents.setContentsMargins(18, 16, 18, 16)
        contents.setSpacing(12)
        heading = QLabel(title)
        heading.setObjectName("helpSection")
        contents.addWidget(heading)
        layout.addWidget(card)
        return contents

    def add_text(self, layout, text):
        label = QLabel(text)
        label.setWordWrap(True)
        layout.addWidget(label)
        return label
