import watchlist from "../watchlist.json";

const VOL = new Set(watchlist.volProducts || []);

function chartUrl(symbol) {
  const encoded = encodeURIComponent(symbol);
  return `https://query1.finance.yahoo.com/v8/finance/chart/${encoded}?interval=1d&range=5d`;
}

async function fetchOne(symbol) {
  const res = await fetch(chartUrl(symbol), {
    headers: {
      "User-Agent": "Mozilla/5.0",
      Accept: "application/json",
    },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Yahoo ${symbol}: HTTP ${res.status}`);
  }
  const json = await res.json();
  const result = json?.chart?.result?.[0];
  if (!result) {
    throw new Error(`Yahoo ${symbol}: empty result`);
  }
  const meta = result.meta || {};
  const price = meta.regularMarketPrice ?? meta.postMarketPrice ?? null;
  const prev = meta.chartPreviousClose ?? meta.previousClose ?? null;
  let change = null;
  let changePct = null;
  if (price != null && prev != null && prev !== 0) {
    change = price - prev;
    changePct = (change / prev) * 100;
  }
  return {
    ticker: symbol,
    price: price == null ? null : Number(price),
    prevClose: prev == null ? null : Number(prev),
    change: change == null ? null : Number(change),
    changePct: changePct == null ? null : Number(changePct),
    currency: meta.currency || null,
    exchange: meta.exchangeName || meta.fullExchangeName || null,
    vol: VOL.has(symbol),
    israel: String(symbol).endsWith(".TA"),
    ok: price != null,
  };
}

export async function fetchWatchlistQuotes() {
  const rows = await Promise.all(
    watchlist.tickers.map(async (ticker) => {
      try {
        return await fetchOne(ticker);
      } catch (error) {
        return {
          ticker,
          ok: false,
          error: error instanceof Error ? error.message : String(error),
          vol: VOL.has(ticker),
          israel: String(ticker).endsWith(".TA"),
        };
      }
    })
  );

  const scored = rows.filter((r) => r.ok && r.changePct != null);
  const gainers = [...scored].sort((a, b) => b.changePct - a.changePct).slice(0, 5);
  const losers = [...scored].sort((a, b) => a.changePct - b.changePct).slice(0, 5);

  return {
    asOf: new Date().toISOString(),
    name: watchlist.name,
    source: "Yahoo Finance public live quotes",
    rows,
    movers: { gainers, losers },
    okCount: rows.filter((r) => r.ok).length,
    total: rows.length,
  };
}
