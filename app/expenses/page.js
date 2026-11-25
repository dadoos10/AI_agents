"use client";

import { useCallback, useMemo, useState } from "react";
import * as XLSX from "xlsx";
import Nav from "../components/Nav";
import { parseMaxMatrix } from "../../lib/maxParse";
import "../globals.css";

function money(n, currency = "ILS") {
  if (n == null || Number.isNaN(n)) return "—";
  try {
    return new Intl.NumberFormat("he-IL", {
      style: "currency",
      currency: currency === "₪" ? "ILS" : currency || "ILS",
      maximumFractionDigits: 2,
    }).format(n);
  } catch {
    return `${Number(n).toLocaleString("he-IL", { maximumFractionDigits: 2 })} ₪`;
  }
}

export default function ExpensesPage() {
  const [dragOver, setDragOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [fileName, setFileName] = useState("");
  const [report, setReport] = useState(null);
  const [query, setQuery] = useState("");

  const ingestFile = useCallback(async (file) => {
    if (!file) return;
    setBusy(true);
    setError("");
    setReport(null);
    setFileName(file.name);
    try {
      const buf = await file.arrayBuffer();
      const wb = XLSX.read(buf, { type: "array", cellDates: true });
      const sheetName = wb.SheetNames[0];
      if (!sheetName) throw new Error("לא נמצא גיליון בקובץ");
      const sheet = wb.Sheets[sheetName];
      const matrix = XLSX.utils.sheet_to_json(sheet, { header: 1, defval: "" });
      const parsed = parseMaxMatrix(matrix, file.name);
      setReport(parsed);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }, []);

  const onInput = (e) => {
    const f = e.target.files?.[0];
    if (f) ingestFile(f);
  };

  const loadSample = async () => {
    setBusy(true);
    setError("");
    try {
      const res = await fetch("/samples/max-sample.csv");
      const text = await res.text();
      const wb = XLSX.read(text, { type: "string" });
      const sheet = wb.Sheets[wb.SheetNames[0]];
      const matrix = XLSX.utils.sheet_to_json(sheet, { header: 1, defval: "" });
      const parsed = parseMaxMatrix(matrix, "max-sample.csv");
      setFileName("max-sample.csv (דוגמה)");
      setReport(parsed);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const filtered = useMemo(() => {
    if (!report) return [];
    const q = query.trim().toLowerCase();
    if (!q) return report.transactions;
    return report.transactions.filter((t) =>
      [t.merchant, t.category, t.type, t.notes, t.date].join(" ").toLowerCase().includes(q)
    );
  }, [report, query]);

  return (
    <main className="app">
      <Nav />

      <header className="hero">
        <div>
          <p className="eyebrow">Max · פירוט חיובים</p>
          <h1>העלאת ייצוא האשראי</h1>
          <p className="sub">
            מייצאים מאקסל באתר/אפליקציית Max ← מעלים כאן. הקובץ נקרא <strong>רק בדפדפן שלך</strong> ולא
            נשלח לשרת.
          </p>
        </div>
      </header>

      <section className="steps card">
        <h2>איך מייצאים מ־Max</h2>
        <ol className="step-list">
          <li>נכנסים ל־max.co.il או לאפליקציה</li>
          <li>עוברים ל־<strong>פירוט חיובים</strong></li>
          <li>בוחרים חודש ולוחצים <strong>ייצוא לאקסל</strong></li>
          <li>מעלים את הקובץ כאן</li>
        </ol>
      </section>

      <section
        className={`dropzone ${dragOver ? "over" : ""} ${busy ? "busy" : ""}`}
        onDragEnter={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          const f = e.dataTransfer.files?.[0];
          if (f) ingestFile(f);
        }}
      >
        <div className="drop-illu">📄</div>
        <h2>גרור לכאן קובץ Excel / CSV</h2>
        <p className="sub">תומך ב־.xlsx · .xls · .csv</p>
        <div className="actions">
          <label className="btn primary">
            בחירת קובץ
            <input type="file" accept=".xlsx,.xls,.csv,text/csv" hidden onChange={onInput} />
          </label>
          <button type="button" className="btn" onClick={loadSample} disabled={busy}>
            טען קובץ לדוגמה
          </button>
        </div>
        {busy ? <p className="sub">מעבד…</p> : null}
        {fileName ? <p className="file-chip">קובץ: {fileName}</p> : null}
      </section>

      {error ? <div className="note danger">{error}</div> : null}

      {report ? (
        <>
          <div className="stats">
            <div className="stat">
              <div className="label">עסקאות</div>
              <div className="value">{report.count}</div>
            </div>
            <div className="stat">
              <div className="label">סה״כ חיובים</div>
              <div className="value">{money(report.total, report.currency)}</div>
            </div>
            <div className="stat">
              <div className="label">קטגוריה מובילה</div>
              <div className="value small">
                {report.topCategories[0]
                  ? `${report.topCategories[0].name}`
                  : "—"}
              </div>
            </div>
            <div className="stat">
              <div className="label">עסק מוביל</div>
              <div className="value small">
                {report.topMerchants[0] ? report.topMerchants[0].name : "—"}
              </div>
            </div>
          </div>

          <div className="movers">
            <section className="card">
              <h2>לפי קטגוריה</h2>
              {report.topCategories.map((c) => (
                <div className="chip" key={c.name}>
                  <span>{c.name}</span>
                  <span className="num">{money(c.value, report.currency)}</span>
                </div>
              ))}
            </section>
            <section className="card">
              <h2>לפי בית עסק</h2>
              {report.topMerchants.map((m) => (
                <div className="chip" key={m.name}>
                  <span>{m.name}</span>
                  <span className="num">{money(m.value, report.currency)}</span>
                </div>
              ))}
            </section>
          </div>

          <section className="card table-wrap">
            <div className="table-head">
              <h2>עסקאות</h2>
              <input
                className="search"
                placeholder="חיפוש עסק / קטגוריה…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>
            <table>
              <thead>
                <tr>
                  <th>תאריך</th>
                  <th>עסק</th>
                  <th>קטגוריה</th>
                  <th className="num">סכום</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((t) => (
                  <tr key={t.id}>
                    <td className="num">{t.date || "—"}</td>
                    <td>
                      <strong>{t.merchant}</strong>
                      {t.type ? <div className="muted">{t.type}</div> : null}
                    </td>
                    <td>{t.category}</td>
                    <td className={`num ${t.amount < 0 ? "up" : ""}`}>
                      {money(t.amount, t.currency)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </>
      ) : null}

      <p className="footer">
        פרטיות: העיבוד מקומי בדפדפן. אין זו ייעוץ פיננסי. פורמט Max עשוי להשתנות — אם הקובץ לא נקרא,
        שלח צילום של שורת הכותרות (בלי סכומים) ונתאים את הפרסר.
      </p>
    </main>
  );
}
