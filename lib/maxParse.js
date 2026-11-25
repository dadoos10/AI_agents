/** Flexible Max / Israeli credit-card Excel-CSV parser (browser-side). */

const HEADER_ALIASES = {
  date: ["תאריך עסקה", "תאריך העסקה", "תאריך", "date", "transaction date", "עסקה בתאריך"],
  billingDate: ["תאריך חיוב", "מועד חיוב", "billing date", "תאריך חיוב בחשבון"],
  merchant: ["שם בית העסק", "בית עסק", "בית העסק", "עסק", "תיאור", "merchant", "שם העסק"],
  category: ["קטגוריה", "ענף", "סיווג", "category"],
  amount: ["סכום חיוב", "סכום", "סכום העסקה", "amount", "חיוב", "זכות"],
  currency: ["מטבע", "currency", "מטבע מקורי"],
  type: ["סוג עסקה", "סוג", "type", "סוג חיוב"],
  notes: ["הערות", "הערה", "פירוט נוסף", "notes", "פרטים"],
  card: ["כרטיס", "4 ספרות", "card", "מספר כרטיס"],
};

function norm(v) {
  return String(v ?? "")
    .replace(/\u200f|\u200e/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}

function scoreHeader(cell, aliases) {
  const n = norm(cell);
  if (!n) return 0;
  for (const a of aliases) {
    const al = norm(a);
    if (n === al) return 100;
    if (n.includes(al) || al.includes(n)) return 80;
  }
  return 0;
}

function detectHeaderRow(matrix) {
  let best = { row: 0, score: -1, map: {} };
  const maxScan = Math.min(matrix.length, 30);
  for (let r = 0; r < maxScan; r++) {
    const row = matrix[r] || [];
    const map = {};
    let score = 0;
    for (let c = 0; c < row.length; c++) {
      for (const [field, aliases] of Object.entries(HEADER_ALIASES)) {
        const s = scoreHeader(row[c], aliases);
        if (s > 0 && (!map[field] || s > map[field].score)) {
          map[field] = { index: c, score: s };
        }
      }
    }
    for (const m of Object.values(map)) score += m.score;
    // Prefer rows that at least found amount + (date or merchant)
    const hasCore = map.amount && (map.date || map.merchant);
    if (hasCore && score > best.score) best = { row: r, score, map };
  }
  if (best.score < 0 || !best.map.amount) {
    throw new Error("לא מצאתי שורת כותרות של Max. ודא שזה ייצוא מפירוט חיובים (אקסל/CSV).");
  }
  return best;
}

function parseAmount(raw) {
  if (raw == null || raw === "") return null;
  if (typeof raw === "number") return raw;
  let s = String(raw).replace(/[₪,\s]/g, "").replace(/−/g, "-");
  // Hebrew sometimes uses parentheses for credit
  const paren = /^\((.+)\)$/.exec(s);
  if (paren) s = `-${paren[1]}`;
  s = s.replace(/[^\d.\-]/g, "");
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

function parseDate(raw) {
  if (raw == null || raw === "") return null;
  if (raw instanceof Date && !Number.isNaN(raw.getTime())) return raw.toISOString().slice(0, 10);
  if (typeof raw === "number") {
    // Excel serial date
    const epoch = Date.UTC(1899, 11, 30);
    const d = new Date(epoch + raw * 86400000);
    if (!Number.isNaN(d.getTime())) return d.toISOString().slice(0, 10);
  }
  const s = String(raw).trim();
  const m = /^(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})$/.exec(s);
  if (m) {
    let [, dd, mm, yy] = m;
    if (yy.length === 2) yy = `20${yy}`;
    const iso = `${yy.padStart(4, "0")}-${mm.padStart(2, "0")}-${dd.padStart(2, "0")}`;
    return iso;
  }
  const d = new Date(s);
  if (!Number.isNaN(d.getTime())) return d.toISOString().slice(0, 10);
  return s;
}

function cell(row, map, field) {
  if (!map[field]) return "";
  return row[map[field].index];
}

/**
 * @param {any[][]} matrix
 * @param {string} fileName
 */
export function parseMaxMatrix(matrix, fileName = "upload") {
  if (!matrix?.length) throw new Error("הקובץ ריק");
  const { row: headerRow, map } = detectHeaderRow(matrix);
  const transactions = [];
  for (let r = headerRow + 1; r < matrix.length; r++) {
    const row = matrix[r] || [];
    if (row.every((c) => c == null || String(c).trim() === "")) continue;
    const amount = parseAmount(cell(row, map, "amount"));
    if (amount == null) continue;
    const merchant = String(cell(row, map, "merchant") ?? "").trim();
    const date = parseDate(cell(row, map, "date"));
    // Skip totals-looking rows
    if (/סה.?כ|total/i.test(merchant)) continue;
    transactions.push({
      id: `${fileName}-${r}-${date}-${amount}-${merchant}`,
      date,
      billingDate: parseDate(cell(row, map, "billingDate")),
      merchant: merchant || "ללא שם",
      category: String(cell(row, map, "category") ?? "").trim() || "כללי",
      amount,
      currency: String(cell(row, map, "currency") ?? "ILS").trim() || "ILS",
      type: String(cell(row, map, "type") ?? "").trim(),
      notes: String(cell(row, map, "notes") ?? "").trim(),
      card: String(cell(row, map, "card") ?? "").trim(),
      sourceFile: fileName,
    });
  }
  if (!transactions.length) {
    throw new Error("נמצאו כותרות, אבל לא נמצאו שורות עסקאות. בדוק שבחרת את חודש החיוב הנכון בייצוא.");
  }

  const total = transactions.reduce((s, t) => s + t.amount, 0);
  const byCategory = {};
  for (const t of transactions) {
    byCategory[t.category] = (byCategory[t.category] || 0) + t.amount;
  }
  const topCategories = Object.entries(byCategory)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 8)
    .map(([name, value]) => ({ name, value }));

  const byMerchant = {};
  for (const t of transactions) {
    byMerchant[t.merchant] = (byMerchant[t.merchant] || 0) + t.amount;
  }
  const topMerchants = Object.entries(byMerchant)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 8)
    .map(([name, value]) => ({ name, value }));

  return {
    fileName,
    count: transactions.length,
    total,
    currency: transactions[0]?.currency || "ILS",
    topCategories,
    topMerchants,
    transactions: transactions.sort((a, b) => String(b.date).localeCompare(String(a.date))),
  };
}
