from flask import Flask, render_template, request, Response, redirect
import sqlite3
import camera

app = Flask(__name__)
DB = "database/attendance.db"


# ---------- INIT DATABASE ----------
def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            date TEXT,
            time TEXT,
            status TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()


# ---------- ROUTES ----------
@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        camera.gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/register", methods=["POST"])
def register():
    camera.STUDENT_NAME = request.form["name"]
    camera.MODE = "register"
    camera.COUNT = 0
    camera.MESSAGE = "Registering..."
    return ("", 204)


@app.route("/attendance")
def attendance():
    camera.MODE = "attendance"
    camera.MESSAGE = "Scanning..."
    return ("", 204)


@app.route("/end_attendance")
def end_attendance():
    camera.MODE = "idle"
    camera.MESSAGE = "Attendance ended"

    camera.mark_absent_remaining()
    return ("", 204)


@app.route("/records")
def records():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT name, date, time, status, id
        FROM attendance
        ORDER BY date DESC, time DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return render_template("attendance.html", records=rows)


@app.route("/edit/<int:record_id>")
def edit_record(record_id):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, name, date, time, status
        FROM attendance
        WHERE id=?
    """, (record_id,))
    record = cur.fetchone()
    conn.close()
    return render_template("edit_attendance.html", record=record)


@app.route("/update", methods=["POST"])
def update_record():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""
        UPDATE attendance
        SET date=?, time=?, status=?
        WHERE id=?
    """, (
        request.form["date"],
        request.form["time"],
        request.form["status"],
        request.form["id"]
    ))
    conn.commit()
    conn.close()
    return redirect("/records")


@app.route("/delete/<int:record_id>", methods=["POST"])
def delete_record(record_id):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("DELETE FROM attendance WHERE id=?", (record_id,))
    conn.commit()
    conn.close()
    return redirect("/records")


if __name__ == "__main__":
    app.run(debug=True, threaded=True)