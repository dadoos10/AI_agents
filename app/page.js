"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Nav from "./components/Nav";
import "./globals.css";

function fmt(n, digits = 2) {
  if (n == null || Number.isNaN(n)) return "—";
  return Number(n).toLocaleString("he-IL", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function pct(n) {
  if (n == null || Number.isNaN(n)) return "—";
  const sign = n > 0 ? "+" : "";
  return `${sign}${fmt(n)}%`;
}

function cls(n) {
  if (n > 0) return "up";
  if (n < 0) return "down";
  return "";
}

export default function HomePage() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState("מתחבר…");

  const refresh = useCallback(async () => {
    try {
      setStatus("מעדכן…");
      const res = await fetch(`/api/quotes?ts=${Date.now()}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      if (json.error) throw new Error(json.error);
      setData(json);
      setError("");
      setStatus("LIVE");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStatus("מנסה שוב");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 30000);
    return () => clearInterval(id);
  }, [refresh]);

  const upCount = useMemo(
    () => (data?.rows || []).filter((r) => (r.changePct ?? 0) > 0).length,
    [data]
  );
  const downCount = useMemo(
    () => (data?.rows || []).filter((r) => (r.changePct ?? 0) < 0).length,
    [data]
  );

  return (
    <main className="app">
      <Nav />
      <div className="header">
        <div>
          <h1>MY LIST · מעקב חי</h1>
          <div className="sub">
            מחירי Yahoo Finance מתעדכנים אוטומטית כל 30 שניות
            {data?.asOf ? ` · עודכן ${new Date(data.asOf).toLocaleString("he-IL")}` : ""}
          </div>
        </div>
        <div className="badge">
          <span className="dot" />
          <span>{status}</span>
        </div>
      </div>

      <div className="note">
        זו אפליקציית ווב חיה לטלפון ולמחשב. הוסף למסך הבית מהדפדפן כדי שתרגיש כמו אפליקציה.
        הנתונים הם ציטוטי שוק ציבוריים — לא התחברות לחשבון Yahoo אישי.
      </div>

      <div className="actions">
        <button className="btn" onClick={refresh} disabled={loading && !data}>
          רענון עכשיו
        </button>
      </div>

      {error ? <div className="note">שגיאה בטעינה: {error}</div> : null}

      <div className="stats">
        <div className="stat">
          <div className="label">טיקרים חיים</div>
          <div className="value">{data ? `${data.okCount}/${data.total}` : "—"}</div>
        </div>
        <div className="stat">
          <div className="label">עולים</div>
          <div className="value up">{data ? upCount : "—"}</div>
        </div>
        <div className="stat">
          <div className="label">יורדים</div>
          <div className="value down">{data ? downCount : "—"}</div>
        </div>
        <div className="stat">
          <div className="label">מוביל</div>
          <div className="value up">
            {data?.movers?.gainers?.[0]
              ? `${data.movers.gainers[0].ticker} ${pct(data.movers.gainers[0].changePct)}`
              : "—"}
          </div>
        </div>
      </div>

      <div className="movers">
        <section className="card">
          <h2>TOP GAINERS</h2>
          {(data?.movers?.gainers || []).map((r) => (
            <div className="chip" key={`g-${r.ticker}`}>
              <span>{r.ticker}</span>
              <span className="up num">{pct(r.changePct)}</span>
            </div>
          ))}
        </section>
        <section className="card">
          <h2>TOP LOSERS</h2>
          {(data?.movers?.losers || []).map((r) => (
            <div className="chip" key={`l-${r.ticker}`}>
              <span>{r.ticker}</span>
              <span className="down num">{pct(r.changePct)}</span>
            </div>
          ))}
        </section>
      </div>

      <section className="card table-wrap">
        <h2>WATCHLIST</h2>
        <table>
          <thead>
            <tr>
              <th>טיקר</th>
              <th>בורסה</th>
              <th className="num">מחיר</th>
              <th className="num">שינוי %</th>
            </tr>
          </thead>
          <tbody>
            {(data?.rows || []).map((r) => (
              <tr key={r.ticker}>
                <td>
                  <strong>{r.ticker}</strong>
                  {r.vol ? <span className="tag">VOL</span> : null}
                  {r.israel ? <span className="tag">TA</span> : null}
                </td>
                <td>{r.exchange || "—"}</td>
                <td className="num">
                  {fmt(r.price, (r.price ?? 0) >= 1000 ? 0 : 2)}
                  {r.currency ? ` ${r.currency}` : ""}
                </td>
                <td className={`num ${cls(r.changePct)}`}>{pct(r.changePct)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <p className="footer">
        אין זו ייעוץ השקעות. מוצרי תנודתיות (SQQQ / UVXY / VIX) מסומנים ב־VOL ומיועדים למסחר קצר־טווח.
        באייפון: Safari → Share → Add to Home Screen.
      </p>
    </main>
  );
}
