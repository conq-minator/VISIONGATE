"""
interface/qr_manager.py
========================
VisionGate Admin Panel – full Tkinter GUI
==========================================
Tabs:
  1  Dashboard       – system overview stats
  2  Users           – browse, add, delete users
  3  QR Manager      – search user → generate / view / save / copy QR
  4  Access Logs     – paginated log viewer with filter
  5  Camera Control  – start/stop camera streams, view active FPS
  6  Dataset Viewer  – browse face sample images per user

Launch standalone:
    python interface/qr_manager.py

Or from main.py as a background thread (non-blocking).
"""

import sys
import os
import io
import threading
import shutil
import logging
import subprocess

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog

# PIL / Pillow for QR image rendering inside Tkinter
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# qrcode
try:
    import qrcode
    from qrcode.image.pil import PilImage
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

import config
from database.db import initialize as init_db
from database.face_storage import (
    list_users, search_users, add_user, delete_user,
    count_face_samples, get_last_access,
    update_user_qr, get_recent_logs,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────
BG          = "#1a1a2e"   # deep navy
PANEL       = "#16213e"   # panel bg
CARD        = "#0f3460"   # card bg
ACCENT      = "#e94560"   # red-pink accent
ACCENT2     = "#533483"   # purple accent
TEXT        = "#eaeaea"   # primary text
TEXT_DIM    = "#8892a0"   # secondary text
SUCCESS     = "#2ecc71"
WARNING     = "#f39c12"
DANGER      = "#e74c3c"
BORDER      = "#2a2a4a"

FONT_TITLE  = ("Segoe UI", 18, "bold")
FONT_HEADER = ("Segoe UI", 12, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_MONO   = ("Consolas", 9)
FONT_SMALL  = ("Segoe UI", 8)


# ─────────────────────────────────────────────────────────────────────────────
# Helper widgets
# ─────────────────────────────────────────────────────────────────────────────

class StyledButton(tk.Button):
    """A rounded-looking button with hover effect."""

    def __init__(self, parent, text, command=None, style="accent", **kw):
        colors = {
            "accent"  : (ACCENT,  "#ff6b7a"),
            "primary" : (CARD,    "#1a527a"),
            "success" : (SUCCESS, "#27ae60"),
            "danger"  : (DANGER,  "#c0392b"),
            "purple"  : (ACCENT2, "#6a44a5"),
        }
        bg_normal, bg_hover = colors.get(style, colors["accent"])
        super().__init__(
            parent,
            text=text,
            command=command,
            bg=bg_normal,
            fg=TEXT,
            activebackground=bg_hover,
            activeforeground=TEXT,
            font=FONT_BODY,
            relief="flat",
            bd=0,
            padx=12,
            pady=6,
            cursor="hand2",
            **kw,
        )
        self._bg = bg_normal
        self._hover = bg_hover
        self.bind("<Enter>", lambda e: self.config(bg=self._hover))
        self.bind("<Leave>", lambda e: self.config(bg=self._bg))


class StatusBar(tk.Label):
    def __init__(self, parent, **kw):
        super().__init__(
            parent,
            text="Ready",
            bg=PANEL,
            fg=TEXT_DIM,
            font=FONT_SMALL,
            anchor="w",
            padx=8,
            **kw,
        )

    def set(self, msg: str, color: str = TEXT_DIM):
        self.config(text=msg, fg=color)


class SectionLabel(tk.Label):
    def __init__(self, parent, text, **kw):
        super().__init__(
            parent,
            text=text,
            bg=PANEL,
            fg=ACCENT,
            font=FONT_HEADER,
            anchor="w",
            **kw,
        )


class InfoRow(tk.Frame):
    """A label-value pair row."""
    def __init__(self, parent, label: str, value: str = "—", **kw):
        super().__init__(parent, bg=CARD, **kw)
        tk.Label(self, text=label, bg=CARD, fg=TEXT_DIM,
                 font=FONT_SMALL, width=18, anchor="w").pack(side="left", padx=(8, 2), pady=4)
        self._val = tk.Label(self, text=value, bg=CARD, fg=TEXT, font=FONT_BODY, anchor="w")
        self._val.pack(side="left", padx=4)

    def update_value(self, value: str):
        self._val.config(text=str(value))


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Dashboard
# ─────────────────────────────────────────────────────────────────────────────

class DashboardTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=PANEL)
        self._build()
        self.refresh()

    def _build(self):
        tk.Label(self, text="System Overview", bg=PANEL, fg=TEXT,
                 font=FONT_TITLE).pack(pady=(20, 4), padx=20, anchor="w")
        tk.Label(self, text="VisionGate Admin Panel", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_BODY).pack(padx=20, anchor="w")

        # Stats grid
        grid = tk.Frame(self, bg=PANEL)
        grid.pack(padx=20, pady=20, fill="x")

        self._stat_frames = {}
        stats = [
            ("total_users",   "Total Users",    "0",  SUCCESS),
            ("face_samples",  "Face Samples",   "0",  ACCENT2),
            ("total_logs",    "Access Logs",    "0",  ACCENT),
            ("model_status",  "Model Status",   "?",  WARNING),
        ]
        for col, (key, label, val, color) in enumerate(stats):
            card = tk.Frame(grid, bg=CARD, padx=16, pady=16, relief="flat")
            card.grid(row=0, column=col, padx=8, sticky="nsew")
            grid.columnconfigure(col, weight=1)

            tk.Label(card, text=label, bg=CARD, fg=TEXT_DIM, font=FONT_SMALL).pack(anchor="w")
            val_lbl = tk.Label(card, text=val, bg=CARD, fg=color, font=("Segoe UI", 22, "bold"))
            val_lbl.pack(anchor="w", pady=(4, 0))
            self._stat_frames[key] = val_lbl

        # Recent log preview
        SectionLabel(self, text="Recent Access Events").pack(padx=20, pady=(10, 4), anchor="w")
        cols = ("Time", "User ID", "Camera", "Decision")
        self._log_tree = self._make_tree(cols)
        self._log_tree.pack(padx=20, pady=4, fill="both", expand=True)

        StyledButton(self, "↻  Refresh", self.refresh, style="primary").pack(
            padx=20, pady=10, anchor="w"
        )

    def _make_tree(self, cols):
        frame  = tk.Frame(self, bg=PANEL)
        style  = _make_treeview_style()
        tree   = ttk.Treeview(frame, columns=cols, show="headings",
                               style="VG.Treeview", height=8)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=140, anchor="center")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        return frame

    def refresh(self):
        init_db()
        users  = list_users()
        logs   = get_recent_logs(limit=200)

        # Count total face samples
        total_samples = sum(count_face_samples(u.user_id) for u in users)
        model_exists  = os.path.isfile(config.MODEL_PATH)

        self._stat_frames["total_users"].config(text=str(len(users)))
        self._stat_frames["face_samples"].config(text=str(total_samples))
        self._stat_frames["total_logs"].config(text=str(len(logs)))
        self._stat_frames["model_status"].config(
            text="Trained" if model_exists else "Not trained",
            fg=SUCCESS if model_exists else DANGER,
        )

        # Refresh log tree
        tree_widget = None
        for w in self._log_tree.winfo_children():
            if isinstance(w, ttk.Treeview):
                tree_widget = w
                break
        if tree_widget:
            tree_widget.delete(*tree_widget.get_children())
            for log in logs[:50]:
                dec = log.decision
                tag = "grant" if "GRANT" in dec else ("qr" if "QR" in dec else "deny")
                tree_widget.insert("", "end",
                    values=(log.timestamp, log.user_id or "—", log.camera_id, dec),
                    tags=(tag,),
                )
            tree_widget.tag_configure("grant", foreground=SUCCESS)
            tree_widget.tag_configure("qr",    foreground=WARNING)
            tree_widget.tag_configure("deny",  foreground=DANGER)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Users
# ─────────────────────────────────────────────────────────────────────────────

class UsersTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=PANEL)
        self._selected_id = None
        self._build()
        self.refresh()

    def _build(self):
        tk.Label(self, text="User Management", bg=PANEL, fg=TEXT,
                 font=FONT_TITLE).pack(pady=(20, 4), padx=20, anchor="w")

        # Toolbar
        bar = tk.Frame(self, bg=PANEL)
        bar.pack(padx=20, pady=8, fill="x")
        StyledButton(bar, "＋ Add User",   self._add_user,    style="success").pack(side="left", padx=4)
        StyledButton(bar, "🗑 Delete",     self._delete_user, style="danger").pack(side="left", padx=4)
        StyledButton(bar, "↻ Refresh",    self.refresh,      style="primary").pack(side="left", padx=4)
        StyledButton(bar, "📟 QR Manager", self._open_qr,    style="purple").pack(side="left", padx=4)

        # Tree
        cols = ("ID", "Name", "QR Code", "Samples", "Last Access")
        frame = tk.Frame(self, bg=PANEL)
        frame.pack(padx=20, pady=4, fill="both", expand=True)
        _make_treeview_style()
        self._tree = ttk.Treeview(frame, columns=cols, show="headings",
                                   style="VG.Treeview", selectmode="browse")
        widths = {"ID": 50, "Name": 180, "QR Code": 160, "Samples": 80, "Last Access": 160}
        for c in cols:
            self._tree.heading(c, text=c, command=lambda col=c: self._sort(col))
            self._tree.column(c, width=widths.get(c, 120), anchor="center")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self._tree.bind("<<TreeviewSelect>>", self._on_select)
        self._tree.bind("<Double-1>", lambda e: self._open_qr())

    def refresh(self):
        self._tree.delete(*self._tree.get_children())
        users = list_users()
        for u in users:
            samples    = count_face_samples(u.user_id)
            last_access = get_last_access(u.user_id) or "Never"
            self._tree.insert("", "end",
                iid=str(u.user_id),
                values=(u.user_id, u.name, u.qr_code or "—", samples, last_access),
            )

    def _on_select(self, _event):
        sel = self._tree.selection()
        self._selected_id = int(sel[0]) if sel else None

    def _sort(self, col):
        rows = [(self._tree.set(k, col), k) for k in self._tree.get_children("")]
        rows.sort()
        for idx, (_, k) in enumerate(rows):
            self._tree.move(k, "", idx)

    def _add_user(self):
        name = simpledialog.askstring("Add User", "Enter user full name:", parent=self)
        if not name:
            return
        qr = simpledialog.askstring("QR Code",
            "Enter QR payload (leave blank to auto-generate):", parent=self)
        user_id = add_user(name.strip(), qr.strip() if qr else None)
        if not qr:
            # Auto-assign VisionGate:<id> QR
            auto_qr = f"VisionGate:{user_id}"
            update_user_qr(user_id, auto_qr)
        messagebox.showinfo("User Added", f"User '{name}' registered with ID {user_id}.")
        self.refresh()

    def _delete_user(self):
        if self._selected_id is None:
            messagebox.showwarning("No Selection", "Select a user first.")
            return
        confirm = messagebox.askyesno(
            "Confirm Delete",
            f"Delete user ID {self._selected_id}? This cannot be undone.",
        )
        if confirm:
            delete_user(self._selected_id)
            self._selected_id = None
            self.refresh()

    def _open_qr(self):
        # Signal to open QR tab for selected user (handled by AdminPanel)
        if self._selected_id is not None:
            self.event_generate("<<OpenQRFor>>", data=str(self._selected_id))


# ─────────────────────────────────────────────────────────────────────────────
# Tab: QR Manager
# ─────────────────────────────────────────────────────────────────────────────

class QRManagerTab(tk.Frame):
    def __init__(self, parent, status_bar: "StatusBar"):
        super().__init__(parent, bg=PANEL)
        self._status    = status_bar
        self._qr_image  = None   # PIL Image
        self._qr_photo  = None   # Tk PhotoImage (must keep ref)
        self._current_user = None
        self._build()

    def _build(self):
        tk.Label(self, text="QR Code Manager", bg=PANEL, fg=TEXT,
                 font=FONT_TITLE).pack(pady=(20, 4), padx=20, anchor="w")
        tk.Label(self, text="Search for a user to generate and view their QR code.",
                 bg=PANEL, fg=TEXT_DIM, font=FONT_BODY).pack(padx=20, anchor="w")

        # ── Search bar ─────────────────────────────────────────────────────
        search_frame = tk.Frame(self, bg=PANEL)
        search_frame.pack(padx=20, pady=16, fill="x")

        tk.Label(search_frame, text="Search user:", bg=PANEL, fg=TEXT_DIM,
                 font=FONT_BODY).pack(side="left")
        self._search_var = tk.StringVar()
        search_entry = tk.Entry(search_frame, textvariable=self._search_var,
                                bg=CARD, fg=TEXT, insertbackground=TEXT,
                                font=FONT_BODY, relief="flat", bd=6, width=28)
        search_entry.pack(side="left", padx=8)
        search_entry.bind("<Return>", lambda e: self._do_search())
        StyledButton(search_frame, "Search", self._do_search, style="accent").pack(side="left")
        StyledButton(search_frame, "Show All", self._show_all, style="primary").pack(side="left", padx=6)

        # ── Split: results list | QR panel ─────────────────────────────────
        split = tk.Frame(self, bg=PANEL)
        split.pack(padx=20, fill="both", expand=True)

        # Left: result list
        left = tk.Frame(split, bg=PANEL, width=260)
        left.pack(side="left", fill="y", padx=(0, 16))
        left.pack_propagate(False)

        SectionLabel(left, "Results").pack(anchor="w", pady=(0, 4))
        list_frame = tk.Frame(left, bg=CARD)
        list_frame.pack(fill="both", expand=True)
        self._result_list = tk.Listbox(
            list_frame, bg=CARD, fg=TEXT, selectbackground=ACCENT,
            selectforeground=TEXT, font=FONT_BODY, relief="flat",
            borderwidth=0, activestyle="none",
        )
        self._result_list.pack(fill="both", expand=True, padx=2, pady=2)
        self._result_list.bind("<<ListboxSelect>>", self._on_user_select)
        self._users_cache = []  # parallel list to listbox items

        # Right: QR display + info
        right = tk.Frame(split, bg=PANEL)
        right.pack(side="left", fill="both", expand=True)

        # User info card
        SectionLabel(right, "User Information").pack(anchor="w", pady=(0, 4))
        info_card = tk.Frame(right, bg=CARD, relief="flat")
        info_card.pack(fill="x", pady=(0, 12))

        self._info_rows = {}
        for key, label in [
            ("user_id",  "User ID"),
            ("name",     "Full Name"),
            ("qr_code",  "QR Payload"),
            ("samples",  "Face Samples"),
            ("last_seen","Last Access"),
        ]:
            row = InfoRow(info_card, label)
            row.pack(fill="x", padx=2, pady=1)
            self._info_rows[key] = row

        # QR display canvas
        SectionLabel(right, "QR Code").pack(anchor="w", pady=(0, 4))
        qr_outer = tk.Frame(right, bg=CARD, relief="flat")
        qr_outer.pack(fill="x")
        self._qr_canvas = tk.Canvas(
            qr_outer, width=320, height=320,
            bg=CARD, highlightthickness=0,
        )
        self._qr_canvas.pack(pady=16)

        if not QR_AVAILABLE:
            self._qr_canvas.create_text(
                160, 160, text="qrcode library not installed\npip install qrcode[pil]",
                fill=DANGER, font=FONT_BODY, justify="center",
            )
        elif not PIL_AVAILABLE:
            self._qr_canvas.create_text(
                160, 160, text="Pillow not installed\npip install pillow",
                fill=DANGER, font=FONT_BODY, justify="center",
            )
        else:
            self._qr_canvas.create_text(
                160, 160, text="Select a user to\ngenerate QR code",
                fill=TEXT_DIM, font=FONT_BODY, justify="center",
            )

        # Action buttons
        btn_row = tk.Frame(right, bg=PANEL)
        btn_row.pack(pady=12, anchor="w")
        StyledButton(btn_row, "🔄  Regenerate",  self._regenerate_qr, style="accent").pack(side="left", padx=4)
        StyledButton(btn_row, "💾  Save PNG",    self._save_qr,        style="success").pack(side="left", padx=4)
        StyledButton(btn_row, "📋  Copy to Clipboard", self._copy_qr, style="purple").pack(side="left", padx=4)
        StyledButton(btn_row, "🖨  Print / Export",   self._export_qr, style="primary").pack(side="left", padx=4)

    # ── Search ────────────────────────────────────────────────────────────────

    def _do_search(self):
        query = self._search_var.get().strip()
        if not query:
            self._show_all()
            return
        results = search_users(query)
        self._populate_list(results)
        self._status.set(f"Found {len(results)} user(s) for '{query}'.",
                         SUCCESS if results else DANGER)

    def _show_all(self):
        results = list_users()
        self._populate_list(results)
        self._status.set(f"Showing all {len(results)} user(s).")

    def _populate_list(self, users):
        self._result_list.delete(0, "end")
        self._users_cache = users
        for u in users:
            self._result_list.insert("end", f"  #{u.user_id}  {u.name}")

    def load_user(self, user_id: int):
        """Load a specific user directly (called from Users tab)."""
        from database.face_storage import get_user
        user = get_user(user_id)
        if user:
            self._populate_list([user])
            self._result_list.selection_set(0)
            self._select_user(user)

    # ── User selection ────────────────────────────────────────────────────────

    def _on_user_select(self, _event):
        sel = self._result_list.curselection()
        if not sel:
            return
        idx  = sel[0]
        if idx >= len(self._users_cache):
            return
        user = self._users_cache[idx]
        self._select_user(user)

    def _select_user(self, user):
        self._current_user = user
        samples    = count_face_samples(user.user_id)
        last_seen  = get_last_access(user.user_id) or "Never"

        self._info_rows["user_id" ].update_value(str(user.user_id))
        self._info_rows["name"    ].update_value(user.name)
        self._info_rows["qr_code" ].update_value(user.qr_code or "—")
        self._info_rows["samples" ].update_value(str(samples))
        self._info_rows["last_seen"].update_value(last_seen)

        self._generate_qr(user)

    # ── QR Generation ─────────────────────────────────────────────────────────

    def _generate_qr(self, user):
        if not QR_AVAILABLE or not PIL_AVAILABLE:
            return

        payload = user.qr_code or f"VisionGate:{user.user_id}"
        self._info_rows["qr_code"].update_value(payload)

        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=8,
            border=4,
        )
        qr.add_data(payload)
        qr.make(fit=True)

        # Light QR on dark background to match admin panel aesthetic
        img = qr.make_image(
            fill_color="#e94560",   # ACCENT (red-pink dots)
            back_color="#0f3460",   # CARD (navy bg)
        )
        self._qr_image = img.get_image()  # PIL Image

        # Resize to canvas size
        size = 300
        pil_resized = self._qr_image.resize((size, size), Image.NEAREST)
        self._qr_photo = ImageTk.PhotoImage(pil_resized)

        self._qr_canvas.delete("all")
        self._qr_canvas.create_image(10, 10, anchor="nw", image=self._qr_photo)
        self._qr_canvas.config(width=size + 20, height=size + 20)

        self._status.set(f"QR generated for {user.name} (ID {user.user_id}).", SUCCESS)

    def _regenerate_qr(self):
        if self._current_user is None:
            messagebox.showwarning("No User", "Select a user first.", parent=self)
            return
        # Re-generate with fresh payload
        new_payload = f"VisionGate:{self._current_user.user_id}"
        update_user_qr(self._current_user.user_id, new_payload)
        from database.face_storage import get_user
        self._current_user = get_user(self._current_user.user_id)
        self._generate_qr(self._current_user)

    # ── Save / Copy / Export ──────────────────────────────────────────────────

    def _save_qr(self):
        if self._qr_image is None:
            messagebox.showwarning("No QR", "Generate a QR code first.", parent=self)
            return
        os.makedirs(config.QR_DIR, exist_ok=True)
        filename = os.path.join(
            config.QR_DIR, f"qr_user_{self._current_user.user_id}.png"
        )
        save_img  = self._qr_image.resize((500, 500), Image.NEAREST)
        save_img.save(filename)
        self._status.set(f"QR saved to {filename}", SUCCESS)
        messagebox.showinfo("Saved", f"QR code saved:\n{filename}", parent=self)

    def _copy_qr(self):
        """Copy QR as PNG bytes to clipboard (Windows only via PowerShell)."""
        if self._qr_image is None:
            messagebox.showwarning("No QR", "Generate a QR code first.", parent=self)
            return
        # Save to temp file then copy to clipboard via PowerShell
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        save_img = self._qr_image.resize((500, 500), Image.NEAREST)
        save_img.save(tmp.name)
        try:
            ps_cmd = (
                f"Add-Type -AssemblyName System.Windows.Forms; "
                f"[System.Windows.Forms.Clipboard]::SetImage("
                f"[System.Drawing.Image]::FromFile('{tmp.name}'))"
            )
            subprocess.run(
                ["powershell", "-Command", ps_cmd],
                capture_output=True, timeout=5,
            )
            self._status.set("QR image copied to clipboard.", SUCCESS)
            messagebox.showinfo("Copied", "QR image copied to clipboard!", parent=self)
        except Exception as exc:
            self._status.set(f"Clipboard error: {exc}", DANGER)
            messagebox.showerror("Error", str(exc), parent=self)

    def _export_qr(self):
        if self._qr_image is None:
            messagebox.showwarning("No QR", "Generate a QR code first.", parent=self)
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")],
            initialfile=f"qr_user_{self._current_user.user_id}.png",
            parent=self,
        )
        if path:
            save_img = self._qr_image.resize((700, 700), Image.NEAREST)
            save_img.save(path)
            self._status.set(f"Exported to {path}", SUCCESS)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Access Logs
# ─────────────────────────────────────────────────────────────────────────────

class LogsTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=PANEL)
        self._filter_var = tk.StringVar(value="ALL")
        self._build()
        self.refresh()

    def _build(self):
        tk.Label(self, text="Access Logs", bg=PANEL, fg=TEXT,
                 font=FONT_TITLE).pack(pady=(20, 4), padx=20, anchor="w")

        # Filter bar
        bar = tk.Frame(self, bg=PANEL)
        bar.pack(padx=20, pady=8, fill="x")
        tk.Label(bar, text="Filter:", bg=PANEL, fg=TEXT_DIM, font=FONT_BODY).pack(side="left")
        for val in ("ALL", "ACCESS GRANTED", "ACCESS DENIED", "QR"):
            rb = tk.Radiobutton(
                bar, text=val, variable=self._filter_var, value=val,
                command=self.refresh,
                bg=PANEL, fg=TEXT, selectcolor=CARD,
                activebackground=PANEL, font=FONT_SMALL,
            )
            rb.pack(side="left", padx=8)
        StyledButton(bar, "↻ Refresh", self.refresh, style="primary").pack(side="right")

        # Tree
        cols = ("ID", "Timestamp", "User ID", "Camera", "Decision")
        frame = tk.Frame(self, bg=PANEL)
        frame.pack(padx=20, pady=4, fill="both", expand=True)
        _make_treeview_style()
        self._tree = ttk.Treeview(frame, columns=cols, show="headings",
                                   style="VG.Treeview", height=20)
        widths = {"ID": 50, "Timestamp": 160, "User ID": 80, "Camera": 70, "Decision": 220}
        for c in cols:
            self._tree.heading(c, text=c)
            self._tree.column(c, width=widths.get(c, 120), anchor="center")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

    def refresh(self):
        self._tree.delete(*self._tree.get_children())
        logs  = get_recent_logs(limit=200)
        filt  = self._filter_var.get()
        for log in logs:
            if filt != "ALL" and filt not in log.decision:
                continue
            tag = "grant" if "GRANT" in log.decision else ("qr" if "QR" in log.decision else "deny")
            self._tree.insert("", "end",
                values=(log.id, log.timestamp, log.user_id or "—", log.camera_id, log.decision),
                tags=(tag,),
            )
        self._tree.tag_configure("grant", foreground=SUCCESS)
        self._tree.tag_configure("qr",    foreground=WARNING)
        self._tree.tag_configure("deny",  foreground=DANGER)


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Camera Control
# ─────────────────────────────────────────────────────────────────────────────

class CameraTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=PANEL)
        self._build()

    def _build(self):
        tk.Label(self, text="Camera Control", bg=PANEL, fg=TEXT,
                 font=FONT_TITLE).pack(pady=(20, 4), padx=20, anchor="w")
        tk.Label(self, text="Configure camera indices and launch live preview.",
                 bg=PANEL, fg=TEXT_DIM, font=FONT_BODY).pack(padx=20, anchor="w")

        # Config section
        cfg_frame = tk.Frame(self, bg=CARD, relief="flat")
        cfg_frame.pack(padx=20, pady=16, fill="x")

        fields = [
            ("Camera IDs (comma-separated):", ", ".join(str(c) for c in config.CAMERA_IDS)),
            ("Capture Width:",                str(config.CAPTURE_WIDTH)),
            ("Capture Height:",               str(config.CAPTURE_HEIGHT)),
            ("Target FPS:",                   str(config.CAPTURE_FPS)),
        ]
        self._cfg_vars = []
        for row_idx, (label, default) in enumerate(fields):
            tk.Label(cfg_frame, text=label, bg=CARD, fg=TEXT_DIM,
                     font=FONT_SMALL).grid(row=row_idx, column=0, padx=12, pady=6, sticky="w")
            var = tk.StringVar(value=default)
            e   = tk.Entry(cfg_frame, textvariable=var, bg=BG, fg=TEXT,
                           insertbackground=TEXT, font=FONT_BODY, relief="flat", bd=4, width=28)
            e.grid(row=row_idx, column=1, padx=12, pady=6, sticky="w")
            self._cfg_vars.append(var)

        # Detect button
        btn_row = tk.Frame(self, bg=PANEL)
        btn_row.pack(padx=20, pady=8, anchor="w")
        StyledButton(btn_row, "🔍  Detect Cameras",  self._detect,      style="accent").pack(side="left", padx=4)
        StyledButton(btn_row, "▶  Launch main.py",   self._launch_main, style="success").pack(side="left", padx=4)

        SectionLabel(self, text="Detected Cameras").pack(padx=20, pady=(10, 4), anchor="w")
        self._cam_label = tk.Label(self, text="Press 'Detect Cameras' to scan.",
                                    bg=PANEL, fg=TEXT_DIM, font=FONT_BODY)
        self._cam_label.pack(padx=20, anchor="w")

    def _detect(self):
        from core.camera_manager import CameraManager
        self._cam_label.config(text="Scanning… (this may take a few seconds)", fg=WARNING)
        self.update()
        found = CameraManager.detect_available_cameras(max_test=7)
        if found:
            self._cam_label.config(
                text=f"Found cameras at indices: {found}   →  Set Camera IDs above and launch.",
                fg=SUCCESS,
            )
        else:
            self._cam_label.config(text="No cameras detected. Check USB connections.", fg=DANGER)

    def _launch_main(self):
        """Launch main.py in a subprocess so the admin panel stays open."""
        main_py = os.path.join(os.path.dirname(__file__), "..", "main.py")
        subprocess.Popen([sys.executable, main_py],
                         cwd=os.path.dirname(main_py))
        messagebox.showinfo("Launched", "VisionGate started in a new process.\n"
                            "Close that window or press Q to stop it.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Dataset Viewer
# ─────────────────────────────────────────────────────────────────────────────

class DatasetTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=PANEL)
        self._user_var     = tk.StringVar()
        self._photo_refs   = []   # keep refs to prevent GC
        self._build()
        self._refresh_users()

    def _build(self):
        tk.Label(self, text="Dataset Viewer", bg=PANEL, fg=TEXT,
                 font=FONT_TITLE).pack(pady=(20, 4), padx=20, anchor="w")
        tk.Label(self, text="Browse face sample images stored for each user.",
                 bg=PANEL, fg=TEXT_DIM, font=FONT_BODY).pack(padx=20, anchor="w")

        ctrl = tk.Frame(self, bg=PANEL)
        ctrl.pack(padx=20, pady=12, fill="x")
        tk.Label(ctrl, text="User:", bg=PANEL, fg=TEXT_DIM, font=FONT_BODY).pack(side="left")
        self._user_cb = ttk.Combobox(ctrl, textvariable=self._user_var,
                                      state="readonly", width=28, font=FONT_BODY)
        self._user_cb.pack(side="left", padx=8)
        self._user_cb.bind("<<ComboboxSelected>>", lambda e: self._load_images())
        StyledButton(ctrl, "↻ Refresh", self._refresh_users, style="primary").pack(side="left")

        self._count_lbl = tk.Label(self, text="", bg=PANEL, fg=TEXT_DIM, font=FONT_SMALL)
        self._count_lbl.pack(padx=20, anchor="w")

        # Scrollable image grid
        canvas = tk.Canvas(self, bg=PANEL, highlightthickness=0)
        vsb    = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True, padx=20)

        self._grid_inner = tk.Frame(canvas, bg=PANEL)
        self._canvas_window = canvas.create_window((0, 0), window=self._grid_inner, anchor="nw")
        self._grid_inner.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
            lambda e: canvas.itemconfig(self._canvas_window, width=e.width))
        self._canvas = canvas

    def _refresh_users(self):
        users = list_users()
        self._users_map = {f"#{u.user_id}  {u.name}": u.user_id for u in users}
        self._user_cb["values"] = list(self._users_map.keys())

    def _load_images(self):
        # Clear grid
        for w in self._grid_inner.winfo_children():
            w.destroy()
        self._photo_refs.clear()

        key = self._user_var.get()
        if not key:
            return
        user_id = self._users_map[key]
        user_dir = os.path.join(config.FACES_DIR, f"user_{user_id}")

        if not os.path.isdir(user_dir):
            tk.Label(self._grid_inner, text="No images found for this user.",
                     bg=PANEL, fg=TEXT_DIM, font=FONT_BODY).grid(row=0, column=0, padx=10, pady=10)
            return

        imgs = [f for f in sorted(os.listdir(user_dir))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self._count_lbl.config(text=f"{len(imgs)} image(s) stored")

        cols = 6
        thumb = 100
        for idx, fname in enumerate(imgs):
            path = os.path.join(user_dir, fname)
            try:
                pil_img = Image.open(path).resize((thumb, thumb), Image.LANCZOS)
                photo   = ImageTk.PhotoImage(pil_img)
                self._photo_refs.append(photo)
                r, c = divmod(idx, cols)
                cell = tk.Frame(self._grid_inner, bg=CARD, relief="flat")
                cell.grid(row=r, column=c, padx=4, pady=4)
                tk.Label(cell, image=photo, bg=CARD).pack()
                tk.Label(cell, text=fname[:10], bg=CARD, fg=TEXT_DIM,
                         font=FONT_SMALL).pack()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Treeview Style helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_treeview_style() -> str:
    style = ttk.Style()
    style.theme_use("clam")
    style.configure(
        "VG.Treeview",
        background=CARD,
        foreground=TEXT,
        fieldbackground=CARD,
        rowheight=26,
        font=FONT_BODY,
        bordercolor=BORDER,
        relief="flat",
    )
    style.configure(
        "VG.Treeview.Heading",
        background=PANEL,
        foreground=ACCENT,
        font=FONT_HEADER,
        relief="flat",
        borderwidth=0,
    )
    style.map("VG.Treeview",
              background=[("selected", ACCENT)],
              foreground=[("selected", TEXT)])
    return "VG.Treeview"


# ─────────────────────────────────────────────────────────────────────────────
# Main AdminPanel window
# ─────────────────────────────────────────────────────────────────────────────

class AdminPanel(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VisionGate Admin Panel")
        self.geometry("1100x750")
        self.minsize(900, 600)
        self.configure(bg=BG)

        # App icon text (fallback if no ico file)
        try:
            self.iconbitmap(default="")
        except Exception:
            pass

        init_db()
        self._build()

    def _build(self):
        # ── Header bar ────────────────────────────────────────────────────────
        header = tk.Frame(self, bg=ACCENT, height=48)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="  🔐  VisionGate Admin", bg=ACCENT, fg="white",
                 font=("Segoe UI", 14, "bold")).pack(side="left", padx=12)
        tk.Label(header, text="Multi-Camera Face Recognition Access Control",
                 bg=ACCENT, fg="#ffd0d8", font=FONT_SMALL).pack(side="left")

        # ── Status bar ────────────────────────────────────────────────────────
        self._status = StatusBar(self)
        self._status.pack(side="bottom", fill="x")

        # Separator
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", side="bottom")

        # ── Side nav + content area ───────────────────────────────────────────
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        # Side nav
        nav = tk.Frame(body, bg=PANEL, width=160)
        nav.pack(side="left", fill="y")
        nav.pack_propagate(False)

        # Content pane (notebook using manual tab switching for full style control)
        self._content = tk.Frame(body, bg=PANEL)
        self._content.pack(side="left", fill="both", expand=True)

        # Build all tabs
        self._tabs = {}
        tab_defs = [
            ("📊  Dashboard",      "dashboard",  lambda: DashboardTab(self._content)),
            ("👥  Users",          "users",      lambda: UsersTab(self._content)),
            ("📟  QR Manager",     "qr",         lambda: QRManagerTab(self._content, self._status)),
            ("📋  Access Logs",    "logs",       lambda: LogsTab(self._content)),
            ("📷  Camera Control", "cameras",    lambda: CameraTab(self._content)),
            ("🖼  Dataset Viewer", "dataset",    lambda: DatasetTab(self._content)),
        ]
        self._nav_btns = {}
        for label, key, factory in tab_defs:
            btn = tk.Button(
                nav, text=label,
                bg=PANEL, fg=TEXT_DIM,
                activebackground=CARD, activeforeground=TEXT,
                font=FONT_BODY, relief="flat", bd=0,
                anchor="w", padx=12, pady=10,
                cursor="hand2",
                command=lambda k=key, f=factory: self._show_tab(k, f),
            )
            btn.pack(fill="x")
            self._nav_btns[key] = btn

        # Wire QR shortcut from Users tab
        # (Users tab fires <<OpenQRFor>> custom event)
        self.bind("<<OpenQRFor>>", self._handle_open_qr_for)

        # Show dashboard by default
        first_key, _, first_factory = tab_defs[0]
        self._current_key = None
        self._show_tab("dashboard", tab_defs[0][2])

    def _show_tab(self, key: str, factory):
        # Hide current
        if self._current_key and self._current_key in self._tabs:
            self._tabs[self._current_key].pack_forget()

        # Lazy-create tab
        if key not in self._tabs:
            self._tabs[key] = factory()

        self._tabs[key].pack(fill="both", expand=True)
        self._current_key = key

        # Update nav highlight
        for k, btn in self._nav_btns.items():
            btn.config(
                bg=CARD if k == key else PANEL,
                fg=TEXT if k == key else TEXT_DIM,
            )
        self._status.set(f"Tab: {key.replace('_', ' ').title()}")

    def _handle_open_qr_for(self, event):
        """Called when Users tab requests QR view for a specific user."""
        # Navigate to QR tab
        self._show_tab("qr", lambda: QRManagerTab(self._content, self._status))
        try:
            user_id = int(event.data)
            qr_tab  = self._tabs.get("qr")
            if qr_tab and hasattr(qr_tab, "load_user"):
                qr_tab.load_user(user_id)
        except (ValueError, AttributeError):
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def launch():
    """Launch the admin panel in the main thread."""
    app = AdminPanel()
    app.mainloop()


if __name__ == "__main__":
    from utils.logger import setup_logging
    setup_logging()
    launch()
