.import React, { useMemo, useState, useCallback, useEffect } from "react";
import {
  Download, Calendar, DollarSign, ListOrdered, CloudDownload,
  RefreshCw, FolderOpen, CheckSquare, XSquare
} from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, LineChart, Line, Legend
} from "recharts";

// -------------------------------------------------------------
// Design Tokens (Dark Mode Default)
// -------------------------------------------------------------
const colors = {
  bgApp: "bg-slate-900",
  bgGradient: "from-slate-900 to-slate-900",
  surface: "bg-slate-800",
  surfaceAlt: "bg-slate-700",
  hover: "hover:bg-slate-700/70",
  border: "border-slate-700",
  text: "text-slate-100",
  textSoft: "text-slate-400",
  textMuted: "text-slate-500",
  focus: "focus:ring-slate-400/40",
  cardShadow: "",
  gridStroke: "#334155",
  axisText: "#e2e8f0",
  tooltipBg: "#1e293b",
  tooltipBorder: "#334155",
  positive: "#16a34a",
  primary: "#3b82f6",
  primaryAlt: "#2563eb",
  neutralBar: "#0ea5e9"
};

// -------------------------------------------------------------
// UI Primitives (Dark Mode Adapted)
// -------------------------------------------------------------
const Card: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ className = "", children, ...rest }) => (
  <div className={`rounded-xl border ${colors.border} ${colors.surface} ${colors.cardShadow} ${className}`} {...rest}>{children}</div>
);
const CardHeader: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ className = "", children, ...rest }) => (
  <div className={`p-4 pb-0 ${className}`}>{children}</div>
);
const CardTitle: React.FC<React.HTMLAttributes<HTMLHeadingElement>> = ({ className = "", children, ...rest }) => (
  <h3 className={`font-semibold leading-none tracking-tight ${className}`}>{children}</h3>
);
const CardContent: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ className = "", children, ...rest }) => (
  <div className={`p-4 pt-2 ${className}`}>{children}</div>
);
const Separator: React.FC<{ className?: string }> = ({ className = "" }) => <div className={`border-t ${colors.border} my-6 ${className}`} />;

const Badge: React.FC<{ variant?: "secondary" | "outline" | "danger" | "success" | "info"; className?: string }> = ({
  variant = "secondary",
  className = "",
  children
}) => {
  const base = "inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium";
  const styles: Record<string, string> = {
    secondary: "bg-slate-600/60 text-slate-100",
    outline: "border border-slate-500 text-slate-200",
    danger: "bg-red-600/30 text-red-300",
    success: "bg-emerald-600/30 text-emerald-300",
    info: "bg-sky-600/30 text-sky-300"
  };
  return <span className={`${base} ${styles[variant]} ${className}`}>{children}</span>;
};

interface TabsOption { label: string; value: string; }
const Tabs: React.FC<{ value: string; onChange: (v: string) => void; options: TabsOption[]; }> = ({ value, onChange, options }) => (
  <div className="inline-flex rounded-md border border-slate-600 bg-slate-800 p-1 gap-1">
    {options.map(o => (
      <button
        key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          className={`px-3 py-1.5 text-sm rounded-md transition ${
            value === o.value
              ? "bg-slate-200 text-slate-900 shadow"
              : "text-slate-300 hover:bg-slate-700/70"
          }`}
      >
        {o.label}
      </button>
    ))}
  </div>
);

const Button: React.FC<
  React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "ghost" | "danger"; size?: "sm" | "md" }
> = ({ className = "", variant = "primary", size = "md", children, ...rest }) => {
  const variants: Record<string, string> = {
    primary: "bg-slate-200 text-slate-900 hover:bg-white",
    secondary: "bg-slate-600 text-slate-100 hover:bg-slate-500",
    ghost: "text-slate-300 hover:bg-slate-700/70",
    danger: "bg-red-600 text-white hover:bg-red-500"
  };
  const sizes: Record<string, string> = { sm: "px-2 py-1 text-xs", md: "px-3 py-2 text-sm" };
  return (
    <button
      className={`inline-flex items-center gap-2 rounded-md font-medium transition disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 ${colors.focus} ${variants[variant]} ${sizes[size]} ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
};

const Input: React.FC<React.InputHTMLAttributes<HTMLInputElement>> = ({ className = "", ...rest }) => (
  <input
    className={`h-9 rounded-md border border-slate-600 bg-slate-900 text-slate-100 placeholder-slate-500 px-3 text-sm shadow-sm focus:outline-none focus:ring-2 ${colors.focus} ${className}`}
    {...rest}
  />
);

// -------------------------------------------------------------
// Types
// -------------------------------------------------------------
interface TradeRow {
  "Symbol & Name": string;
  "Trade Date": string;
  "Settlement Date"?: string;
  "Buy/Sell": string;
  Quantity?: number | string;
  "Traded Price"?: number | string;
  "Gross Amount"?: number | string;
  "Comm/Fee/Tax"?: number | string;
  VAT?: number | string;
  "Net Amount"?: number | string;
  source_file?: string;
  __date?: Date | null;
  __month?: string;
}
interface MonthlySummary {
  month: string;
  orders: number;
  net_amount_usd: number;
  avg_net_per_order_usd: number;
}
interface MonthlySymbolSummary {
  month: string;
  symbol: string;
  orders: number;
  net_amount_usd: number;
  avg_net_per_order_usd: number;
}
interface RemoteFileMeta {
  name: string;
  path: string;
  size: number;
  download_url: string;
  sha: string;
}

// -------------------------------------------------------------
// Helpers
// -------------------------------------------------------------
function isTruthy<T>(v: T | undefined | null): v is T { return v !== undefined && v !== null; }
function parseNumber(x: unknown): number {
  if (typeof x === "number") return x;
  if (typeof x === "string") {
    const cleaned = x.replace(/[^0-9+\-.,]/g, "").replace(/,(?=\d{3}\b)/g, "");
    const dotComma = cleaned.replace(/,/g, ".");
    const n = Number(dotComma);
    return Number.isFinite(n) ? n : NaN;
  }
  return NaN;
}
function detectHeaderLine(lines: string[]): number {
  for (let i = 0; i < Math.min(80, lines.length); i++) {
    const ln = lines[i];
    if (ln.includes("Symbol & Name") && ln.includes("Trade Date") && ln.includes("Buy/Sell")) return i;
  }
  return 0;
}
function toYMDMonth(d: Date): string {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`;
}
function parseDateFlexible(s: string): Date | null {
  const m = s.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
  if (m) {
    const dt = new Date(Number(m[3]), Number(m[2]) - 1, Number(m[1]));
    return Number.isNaN(dt.getTime()) ? null : dt;
  }
  const nat = new Date(s);
  return Number.isNaN(nat.getTime()) ? null : nat;
}
function parseCSV(text: string): string[][] {
  const rows: string[][] = [];
  let field = "", row: string[] = [], inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; }
        else inQuotes = false;
      } else field += c;
    } else {
      if (c === '"') inQuotes = true;
      else if (c === ",") { row.push(field); field = ""; }
      else if (c === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
      else if (c === "\r") continue;
      else field += c;
    }
  }
  if (field.length > 0 || row.length) { row.push(field); rows.push(row); }
  return rows.filter(r => r.some(c => c.trim() !== ""));
}
function csvToObjects(csvText: string): TradeRow[] {
  const matrix = parseCSV(csvText.replace(/\uFEFF/g, ""));
  if (!matrix.length) return [];
  const header = matrix[0].map(h => h.trim());
  const out: TradeRow[] = [];
  for (let i = 1; i < matrix.length; i++) {
    const cols = matrix[i];
    if (!cols.length) continue;
    const obj: any = {};
    header.forEach((h, idx) => obj[h] = cols[idx]);
    out.push(obj);
  }
  return out;
}
function buildCSV(rows: Record<string, unknown>[]): string {
  if (!rows.length) return "";
  const headers = Object.keys(rows[0]);
  const esc = (v: unknown) => {
    if (v == null) return "";
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [headers.join(","), ...rows.map(r => headers.map(h => esc(r[h])).join(","))].join("\n");
}
function downloadCSV(filename: string, rows: Record<string, unknown>[]) {
  const blob = new Blob([buildCSV(rows)], { type: "text/csv;charset=utf-8;" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}
function parseStatementText(fileName: string, text: string): TradeRow[] {
  const raw = text.split(/\r?\n/);
  const idx = detectHeaderLine(raw);
  return csvToObjects(raw.slice(idx).join("\n")).map(r => ({ ...r, source_file: fileName }));
}

// -------------------------------------------------------------
// Option Detection
// -------------------------------------------------------------
const OPTION_REGEX = /^[A-Z0-9.\-]{1,8}\s+\d{6}[CP]\d{8}$/;
function isOptionContract(symbolNameRaw: string | undefined): boolean {
  if (!symbolNameRaw) return false;
  const s = symbolNameRaw.trim().toUpperCase();
  return OPTION_REGEX.test(s);
}

// -------------------------------------------------------------
// Component
// -------------------------------------------------------------
const Dashboard: React.FC = () => {
  const [rows, setRows] = useState<TradeRow[]>([]);
  const [symbolFilter, setSymbolFilter] = useState("");
  const [tradeType, setTradeType] = useState<"SELL" | "BUY" | "ALL">("SELL");
  const [selectedMonth, setSelectedMonth] = useState<string | null>(null);

  // Active / Skip
  const [fileActiveMap, setFileActiveMap] = useState<Record<string, boolean>>({});
  useEffect(() => {
    const next = { ...fileActiveMap };
    let changed = false;
    for (const f of new Set(rows.map(r => r.source_file || "UNKNOWN"))) {
      if (next[f] === undefined) { next[f] = true; changed = true; }
    }
    if (changed) setFileActiveMap(next);
  }, [rows]);
  const toggleFileActive = (file: string) => setFileActiveMap(prev => ({ ...prev, [file]: !prev[file] }));
  const setAllFiles = (active: boolean) =>
    setFileActiveMap(prev => Object.fromEntries(Object.keys(prev).map(k => [k, active])));

  // Local upload
  const onFileChange: React.ChangeEventHandler<HTMLInputElement> = e => {
    const fs = Array.from(e.target.files ?? []);
    if (!fs.length) return;
    const acc: TradeRow[] = [];
    let done = 0;
    fs.forEach(f => {
      const reader = new FileReader();
      reader.onload = () => {
        acc.push(...parseStatementText(f.name, String(reader.result || "")));
        done++;
        if (done === fs.length) setRows(prev => [...prev, ...acc]);
      };
      reader.readAsText(f);
    });
  };

  // Remote (GitHub)
  const [showRemote, setShowRemote] = useState(false);
  const [remoteLoading, setRemoteLoading] = useState(false);
  const [remoteError, setRemoteError] = useState<string | null>(null);
  const [remoteFiles, setRemoteFiles] = useState<RemoteFileMeta[]>([]);
  const [selectedRemote, setSelectedRemote] = useState<Set<string>>(new Set());
  const [remoteImporting, setRemoteImporting] = useState(false);
  const [remoteImportProgress, setRemoteImportProgress] = useState({ done: 0, total: 0 });

  const toggleRemoteFile = (name: string) => {
    setSelectedRemote(prev => {
      const n = new Set(prev);
      n.has(name) ? n.delete(name) : n.add(name);
      return n;
    });
  };

  const fetchRemoteList = useCallback(async () => {
    setRemoteLoading(true);
    setRemoteError(null);
    setSelectedRemote(new Set());
    try {
      const url = "https://api.github.com/repos/firstnattapon/streamlit-example-1/contents/webull_log?ref=master";
      const res = await fetch(url, { headers: { Accept: "application/vnd.github.v3+json" } });
      if (!res.ok) throw new Error("GitHub API error: " + res.status);
      const data = await res.json();
      if (!Array.isArray(data)) throw new Error("Unexpected response");
      setRemoteFiles(
        data.filter((f: any) => f.type === "file" && /\.csv$/i.test(f.name)).map((f: any) => ({
          name: f.name, path: f.path, size: f.size, download_url: f.download_url, sha: f.sha
        }))
      );
    } catch (e: any) {
      setRemoteError(e.message || "ไม่สามารถดึงรายชื่อไฟล์ได้");
    } finally {
      setRemoteLoading(false);
    }
  }, []);

  const importSelectedRemote = useCallback(async (mode: "append" | "replace" = "append") => {
    if (!selectedRemote.size) return;
    setRemoteImporting(true);
    setRemoteImportProgress({ done: 0, total: selectedRemote.size });
    const collected: TradeRow[] = [];
    let done = 0;
    for (const f of remoteFiles) {
      if (!selectedRemote.has(f.name)) continue;
      try {
        const res = await fetch(f.download_url);
        if (!res.ok) throw new Error("โหลดไฟล์ล้มเหลว: " + f.name);
        const text = await res.text();
        collected.push(...parseStatementText("GH:" + f.name, text));
      } catch (err) { console.warn(err); }
      finally {
        done++;
        setRemoteImportProgress({ done, total: selectedRemote.size });
      }
    }
    setRows(prev => mode === "replace" ? collected : [...prev, ...collected]);
    setRemoteImporting(false);
  }, [remoteFiles, selectedRemote]);

  // File meta
  const fileMeta = useMemo(() => {
    const map = new Map<string, { count: number; active: boolean; sourceType: "GH" | "LOCAL" | "OTHER" }>();
    for (const r of rows) {
      const k = r.source_file || "UNKNOWN";
      const src = k.startsWith("GH:") ? "GH" : (k === "UNKNOWN" ? "OTHER" : "LOCAL");
      const cur = map.get(k);
      if (cur) cur.count++;
      else map.set(k, { count: 1, active: fileActiveMap[k] !== false, sourceType: src });
    }
    return Array.from(map.entries()).map(([file, v]) => ({
      file, count: v.count, active: v.active, sourceType: v.sourceType
    })).sort((a,b)=>a.file.localeCompare(b.file));
  }, [rows, fileActiveMap]);

  // Active rows
  const activeRows = useMemo(
    () => rows.filter(r => fileActiveMap[r.source_file || "UNKNOWN"] !== false),
    [rows, fileActiveMap]
  );

  // Remove Options
  const { equityRows, removedOptionCount } = useMemo(() => {
    let removed = 0;
    const keep: TradeRow[] = [];
    for (const r of activeRows) {
      const sym = r["Symbol & Name"];
      if (isOptionContract(sym) || (sym && /\s\d{6}[CP]\d{6,}/.test(sym.toUpperCase()))) {
        removed++;
        continue;
      }
      keep.push(r);
    }
    return { equityRows: keep, removedOptionCount: removed };
  }, [activeRows]);

  // Filter conditions
  const filtered = useMemo(() => {
    let arr = equityRows.filter(r =>
      r["Buy/Sell"] && (tradeType === "ALL" || r["Buy/Sell"].toUpperCase() === tradeType)
    );
    if (symbolFilter.trim()) {
      const q = symbolFilter.toLowerCase();
      arr = arr.filter(r => r["Symbol & Name"]?.toLowerCase().includes(q));
    }
    return arr.map(r => ({
      ...r,
      Quantity: parseNumber(r.Quantity),
      "Traded Price": parseNumber(r["Traded Price"]),
      "Gross Amount": parseNumber(r["Gross Amount"]),
      "Comm/Fee/Tax": parseNumber(r["Comm/Fee/Tax"]),
      VAT: parseNumber(r.VAT),
      "Net Amount": parseNumber(r["Net Amount"])
    }));
  }, [equityRows, symbolFilter, tradeType]);

  // Enriched dates
  const enriched = useMemo(() => filtered.map(r => {
    const d = r["Trade Date"] ? parseDateFlexible(String(r["Trade Date"])) : null;
    const month = d ? toYMDMonth(d) : "Unknown";
    return { ...r, __date: d, __month: month };
  }).filter(r => isTruthy(r.__date)), [filtered]);

  // Monthly summary
  const monthly: MonthlySummary[] = useMemo(() => {
    const m = new Map<string,{orders:number;net:number}>();
    for (const r of enriched) {
      const key = r.__month as string;
      const net = parseNumber(r["Net Amount"]);
      const cur = m.get(key) ?? { orders: 0, net: 0 };
      m.set(key, { orders: cur.orders + 1, net: cur.net + (Number.isFinite(net) ? net : 0) });
    }
    return Array.from(m.entries()).map(([month,v])=>({
      month,
      orders: v.orders,
      net_amount_usd: v.net,
      avg_net_per_order_usd: v.orders ? v.net / v.orders : 0
    })).sort((a,b)=>a.month.localeCompare(b.month));
  }, [enriched]);

  // Monthly + Symbol
  const monthlyBySymbol: MonthlySymbolSummary[] = useMemo(() => {
    const map = new Map<string,{orders:number;net:number}>();
    for (const r of enriched) {
      const month = r.__month as string;
      const symbol = (r["Symbol & Name"] || "").trim() || "(ไม่ระบุ)";
      const key = month + "||" + symbol;
      const net = parseNumber(r["Net Amount"]);
      const cur = map.get(key) ?? { orders: 0, net: 0 };
      map.set(key, { orders: cur.orders + 1, net: cur.net + (Number.isFinite(net) ? net : 0) });
    }
    const arr: MonthlySymbolSummary[] = [];
    map.forEach((v, key) => {
      const [month, symbol] = key.split("||");
      arr.push({
        month,
        symbol,
        orders: v.orders,
        net_amount_usd: v.net,
        avg_net_per_order_usd: v.orders ? v.net / v.orders : 0
      });
    });
    return arr.sort((a,b) => {
      if (a.month === b.month) return Math.abs(b.net_amount_usd) - Math.abs(a.net_amount_usd);
      return a.month.localeCompare(b.month);
    });
  }, [enriched]);

  // Symbol data for selected month (chart)
  const selectedMonthSymbolData = useMemo(() => {
    if (!selectedMonth) return [];
    return monthlyBySymbol
      .filter(r => r.month === selectedMonth)
      .sort((a,b) => Math.abs(b.net_amount_usd) - Math.abs(a.net_amount_usd));
  }, [monthlyBySymbol, selectedMonth]);

  // Cumulative
  const cumulativeMonthly = useMemo(() => {
    let cum = 0;
    return monthly.map(m => {
      cum += m.net_amount_usd;
      return { month: m.month, cumulative_net_usd: cum };
    });
  }, [monthly]);

  // KPI
  const kpis = useMemo(() => {
    const totalOrders = monthly.reduce((s,m)=>s+m.orders,0);
    const totalNet = monthly.reduce((s,m)=>s+m.net_amount_usd,0);
    const monthlyAvg = monthly.length ? totalNet / monthly.length : 0;
    return { totalOrders, totalNet, monthlyAvg };
  }, [monthly]);

  // Export selected month symbol breakdown
  const exportSelectedMonthSymbols = () => {
    if (!selectedMonth || !selectedMonthSymbolData.length) return;
    downloadCSV(`symbol_breakdown_${selectedMonth}.csv`,
      selectedMonthSymbolData.map(d => ({
        month: d.month,
        symbol: d.symbol,
        orders: d.orders,
        net_amount_usd: d.net_amount_usd,
        avg_net_per_order_usd: d.avg_net_per_order_usd
      }))
    );
  };

  // Tooltip style (dark)
  const tooltipStyle = {
    background: colors.tooltipBg,
    border: `1px solid ${colors.tooltipBorder}`,
    color: "#f1f5f9",
    fontSize: "12px"
  };

  return (
    <div className={`min-h-screen w-full p-6 md:p-10 ${colors.bgApp} text-slate-100`}>
      <div className="max-w-7xl mx-auto space-y-6">

        {/* Header */}
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold tracking-tight">
              Data Dashboard • SELL Monthly (Exclude Options)
            </h1>
            <p className="text-slate-400 text-sm">
              เลือกเดือนจาก "ตารางสรุปรายเดือน" เพื่อดูกราฟมูลค่ารวม (USD) ราย Symbol (Dark Mode)
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <label htmlFor="file" className="sr-only">Upload</label>
            <Input id="file" type="file" multiple accept=".csv" onChange={onFileChange} />
            <Button
              variant="secondary"
              onClick={() => downloadCSV("sell_monthly_summary_by_symbol_no_options.csv", monthlyBySymbol)}
              disabled={!monthlyBySymbol.length}
            >
              <Download className="w-4 h-4" /> ดาวน์โหลด (Month+Symbol)
            </Button>
            <Button
              variant="ghost"
              disabled={!selectedMonthSymbolData.length}
              onClick={exportSelectedMonthSymbols}
            >
              <Download className="w-4 h-4" /> Export เดือนที่เลือก
            </Button>
            <Button variant="ghost" onClick={() => setShowRemote(s => !s)}>
              <CloudDownload className="w-4 h-4" /> {showRemote ? "ซ่อน GitHub" : "นำเข้าจาก GitHub"}
            </Button>
          </div>
        </div>

          {/* Option removal info bar */}
          <Card>
            <CardContent className="py-3 flex flex-wrap gap-4 text-xs items-center">
              <Badge variant="info">Option Filter</Badge>
              <div>ตัด Option ออกแล้ว {removedOptionCount.toLocaleString()} แถว</div>
              <div>แถวหุ้นสำหรับคำนวณ: {equityRows.length.toLocaleString()}</div>
              <div className="text-slate-500">
                Pattern: UNDERLYING + YYMMDD + C/P + Strike 8 ตัว
              </div>
            </CardContent>
          </Card>

        {/* Remote GitHub */}
        {showRemote && (
          <Card className="border-2 border-dashed border-slate-600">
            <CardHeader className="flex flex-col gap-2 pb-2">
              <div className="flex items-center gap-2">
                <FolderOpen className="w-4 h-4 text-slate-300" />
                <CardTitle className="text-sm">Remote GitHub: webull_log (master)</CardTitle>
                <Button size="sm" variant="secondary" onClick={fetchRemoteList} disabled={remoteLoading}>
                  {remoteLoading ? <RefreshCw className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
                  โหลดรายชื่อไฟล์
                </Button>
              </div>
              <div className="text-[11px] text-slate-400">
                เลือกไฟล์ .csv แล้วนำเข้า (Append / Replace)
              </div>
              {remoteError && <div className="text-xs text-red-400">{remoteError}</div>}
            </CardHeader>
            <CardContent className="pt-2 space-y-3">
              <div className="flex flex-wrap gap-2 items-center">
                <Button size="sm" variant="ghost" disabled={!remoteFiles.length}
                        onClick={() => setSelectedRemote(new Set(remoteFiles.map(f => f.name)))}>เลือกทั้งหมด</Button>
                <Button size="sm" variant="ghost" disabled={!selectedRemote.size}
                        onClick={() => setSelectedRemote(new Set())}>ล้างเลือก</Button>
                <Button size="sm" variant="primary" disabled={!selectedRemote.size || remoteImporting}
                        onClick={() => importSelectedRemote("append")}>
                  {remoteImporting ? <RefreshCw className="w-3 h-3 animate-spin" /> : <CloudDownload className="w-3 h-3" />}
                  นำเข้า (Append)
                </Button>
                <Button size="sm" variant="danger" disabled={!selectedRemote.size || remoteImporting}
                        onClick={() => importSelectedRemote("replace")}>
                  {remoteImporting ? <RefreshCw className="w-3 h-3 animate-spin" /> : <CloudDownload className="w-3 h-3" />}
                  นำเข้า (Replace)
                </Button>
                {remoteImporting && (
                  <span className="text-xs text-slate-400">
                    กำลังนำเข้า {remoteImportProgress.done}/{remoteImportProgress.total}
                  </span>
                )}
              </div>
              <div className="max-h-64 overflow-auto rounded border border-slate-600">
                <table className="w-full text-xs">
                  <thead className="bg-slate-700 text-slate-200">
                    <tr>
                      <th className="px-2 py-2 text-left font-medium">เลือก</th>
                      <th className="px-2 py-2 text-left font-medium">ไฟล์</th>
                      <th className="px-2 py-2 text-right font-medium">ขนาด</th>
                      <th className="px-2 py-2 text-left font-medium">SHA</th>
                    </tr>
                  </thead>
                  <tbody>
                    {remoteLoading && (
                      <tr><td colSpan={4} className="px-3 py-4 text-center text-slate-400">กำลังโหลด...</td></tr>
                    )}
                    {!remoteLoading && !remoteFiles.length && (
                      <tr><td colSpan={4} className="px-3 py-4 text-center text-slate-500">ยังไม่มีรายชื่อไฟล์</td></tr>
                    )}
                    {remoteFiles.map(f => {
                      const checked = selectedRemote.has(f.name);
                      return (
                        <tr key={f.sha} className="border-t border-slate-700 hover:bg-slate-700/60">
                          <td className="px-2 py-1">
                            <input type="checkbox" className="cursor-pointer accent-slate-200" checked={checked} onChange={() => toggleRemoteFile(f.name)} />
                          </td>
                          <td className="px-2 py-1 font-medium whitespace-nowrap text-slate-100">{f.name}</td>
                          <td className="px-2 py-1 text-right tabular-nums text-slate-300">{f.size.toLocaleString()}</td>
                          <td className="px-2 py-1 text-[10px] font-mono text-slate-400">{f.sha.slice(0,10)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              {!!rows.length && (
                <div className="text-[11px] text-slate-500">
                  ข้อมูลรวม (ก่อนตัด option): {rows.length.toLocaleString()} แถว
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* File Active Management */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <CheckSquare className="w-4 h-4" /> การจัดการไฟล์ (Active / Skip)
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap gap-2">
              <Button size="sm" variant="ghost" onClick={() => setAllFiles(true)} disabled={!fileMeta.length}>
                <CheckSquare className="w-3 h-3" /> ใช้ทั้งหมด
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setAllFiles(false)} disabled={!fileMeta.length}>
                <XSquare className="w-3 h-3" /> ข้ามทั้งหมด
              </Button>
            </div>
            <div className="max-h-60 overflow-auto rounded border border-slate-600">
              <table className="w-full text-xs">
                <thead className="bg-slate-700 text-slate-300">
                  <tr>
                    <th className="px-2 py-2 text-left font-medium">ไฟล์</th>
                    <th className="px-2 py-2 text-right font-medium">แถว</th>
                    <th className="px-2 py-2 text-left font-medium">ประเภท</th>
                    <th className="px-2 py-2 text-left font-medium">สถานะ</th>
                    <th className="px-2 py-2 text-center font-medium">Active?</th>
                  </tr>
                </thead>
                <tbody>
                  {fileMeta.map(f => (
                    <tr key={f.file} className="border-t border-slate-700 hover:bg-slate-700/60">
                      <td className="px-2 py-1 font-medium truncate max-w-[240px] text-slate-100" title={f.file}>{f.file}</td>
                      <td className="px-2 py-1 text-right text-slate-300">{f.count.toLocaleString()}</td>
                      <td className="px-2 py-1">
                        {f.sourceType === "GH" && <Badge variant="outline">GitHub</Badge>}
                        {f.sourceType === "LOCAL" && <Badge variant="secondary">Local</Badge>}
                        {f.sourceType === "OTHER" && <Badge variant="danger">Unknown</Badge>}
                      </td>
                      <td className="px-2 py-1">
                        {f.active
                          ? <Badge variant="success">สถานะ:ใช้แล้ว</Badge>
                          : <Badge variant="danger">สถานะ:ข้าม</Badge>}
                      </td>
                      <td className="px-2 py-1 text-center">
                        <input type="checkbox" checked={f.active} onChange={() => toggleFileActive(f.file)} className="cursor-pointer accent-slate-200" />
                      </td>
                    </tr>
                  ))}
                  {!fileMeta.length && (
                    <tr>
                      <td colSpan={5} className="px-2 py-6 text-center text-slate-500">
                        ยังไม่มีไฟล์นำเข้า
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
            <div className="text-[11px] text-slate-500">
              ระบบคำนวณเฉพาะไฟล์ที่ Active และเฉพาะหุ้น (Option ถูกตัดออก)
            </div>
          </CardContent>
        </Card>

        {/* KPI Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2 text-slate-300">
                <ListOrdered className="w-4 h-4" /> จำนวนออเดอร์ SELL (ไม่รวม Option)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-semibold text-slate-50">{kpis.totalOrders.toLocaleString()}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2 text-slate-300">
                <DollarSign className="w-4 h-4" /> มูลค่า SELL รวม (USD)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-semibold text-slate-50">
                {kpis.totalNet.toLocaleString(undefined, { maximumFractionDigits: 2 })}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2 text-slate-300">
                <Calendar className="w-4 h-4" /> ค่าเฉลี่ย (USD/เดือน)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-semibold text-slate-50">
                {kpis.monthlyAvg.toLocaleString(undefined, { maximumFractionDigits: 2 })}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Filters + Global Charts + Tables */}
        <Card>
          <CardContent className="pt-6 space-y-6">
            <div className="flex flex-col lg:flex-row lg:items-center gap-4">
              <div className="flex items-center gap-2">
                <Badge variant="secondary">Filter</Badge>
                <Input
                  placeholder="ค้นหา Symbol & Name"
                  value={symbolFilter}
                  onChange={e => setSymbolFilter(e.target.value)}
                  className="w-64"
                />
              </div>
              <div className="flex items-center gap-3">
                <Badge variant="outline">Type</Badge>
                <Tabs
                  value={tradeType}
                  onChange={(v) => setTradeType(v as any)}
                  options={[
                    { label: "SELL", value: "SELL" },
                    { label: "BUY", value: "BUY" },
                    { label: "ALL", value: "ALL" }
                  ]}
                />
              </div>
              <div className="text-xs text-slate-400">
                ใช้งานไฟล์: {fileMeta.filter(f => f.active).length}/{fileMeta.length} ไฟล์
              </div>
            </div>

            {/* Charts (Global) */}
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
              <Card className="border-dashed border-slate-600">
                <CardHeader>
                  <CardTitle className="text-base text-slate-200">จำนวนออเดอร์รายเดือน</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-72">
                    <ResponsiveContainer>
                      <BarChart data={monthly}>
                        <CartesianGrid vertical={false} strokeDasharray="3 3" stroke={colors.gridStroke} />
                        <XAxis dataKey="month" tick={{ fontSize: 12, fill: colors.axisText }} />
                        <YAxis tick={{ fontSize: 12, fill: colors.axisText }} />
                        <Tooltip
                          contentStyle={tooltipStyle as any}
                          formatter={(v:number)=>v.toLocaleString()}
                        />
                        <Bar dataKey="orders" fill={colors.neutralBar} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              <Card className="border-dashed border-slate-600">
                <CardHeader>
                  <CardTitle className="text-base text-slate-200">มูลค่า Net Amount รายเดือน (USD)</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-72">
                    <ResponsiveContainer>
                      <BarChart data={monthly}>
                        <CartesianGrid vertical={false} strokeDasharray="3 3" stroke={colors.gridStroke} />
                        <XAxis dataKey="month" tick={{ fontSize: 12, fill: colors.axisText }} />
                        <YAxis tick={{ fontSize: 12, fill: colors.axisText }} />
                        <Tooltip
                          contentStyle={tooltipStyle as any}
                          formatter={(v:number)=>v.toLocaleString(undefined,{maximumFractionDigits:2})}
                        />
                        <Bar dataKey="net_amount_usd" fill={colors.primaryAlt} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              <Card className="border-dashed border-slate-600">
                <CardHeader>
                  <CardTitle className="text-base text-slate-200">มูลค่าสะสม (USD)</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-72">
                    <ResponsiveContainer>
                      <LineChart data={cumulativeMonthly}>
                        <CartesianGrid vertical={false} strokeDasharray="3 3" stroke={colors.gridStroke} />
                        <XAxis dataKey="month" tick={{ fontSize: 12, fill: colors.axisText }} />
                        <YAxis tick={{ fontSize: 12, fill: colors.axisText }} />
                        <Tooltip
                          contentStyle={tooltipStyle as any}
                          formatter={(v:number)=>v.toLocaleString(undefined,{maximumFractionDigits:2})}
                        />
                        <Legend wrapperStyle={{ color: colors.axisText, fontSize: 12 }} />
                        <Line type="stepAfter" dataKey="cumulative_net_usd" stroke={colors.positive} strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </div>

            <Separator />

            {/* Monthly Summary Table (Selectable) + Symbol Chart */}
            <div className="grid grid-cols-1 2xl:grid-cols-5 gap-6">
              <div className="2xl:col-span-2 flex flex-col gap-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-slate-200">ตารางสรุปรายเดือน</h3>
                  <div className="text-xs text-slate-400">{monthly.length} เดือน</div>
                </div>
                <div className="overflow-auto rounded-lg border border-slate-600 max-h-[520px]">
                  <table className="w-full text-sm">
                    <thead className="bg-slate-700 text-slate-200">
                      <tr>
                        <th className="text-left px-3 py-2 font-medium">เดือน</th>
                        <th className="text-right px-3 py-2 font-medium">Orders</th>
                        <th className="text-right px-3 py-2 font-medium">Net (USD)</th>
                        <th className="text-right px-3 py-2 font-medium">Avg / Order</th>
                      </tr>
                    </thead>
                    <tbody>
                      {monthly.map(m => {
                        const active = selectedMonth === m.month;
                        return (
                          <tr
                            key={m.month}
                            onClick={() => setSelectedMonth(m.month === selectedMonth ? null : m.month)}
                            className={`border-t border-slate-700 cursor-pointer ${colors.hover} ${
                              active ? "bg-slate-700 ring-1 ring-slate-500" : ""
                            }`}
                          >
                            <td className="px-3 py-2 font-medium whitespace-nowrap text-slate-100">
                              {m.month}
                              {selectedMonth === m.month && <span className="ml-2 text-[10px] text-emerald-400">เลือก</span>}
                            </td>
                            <td className="px-3 py-2 text-right text-slate-300">{m.orders.toLocaleString()}</td>
                            <td className="px-3 py-2 text-right text-slate-200">
                              {m.net_amount_usd.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                            </td>
                            <td className="px-3 py-2 text-right text-slate-300">
                              {m.avg_net_per_order_usd.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                            </td>
                          </tr>
                        );
                      })}
                      {!monthly.length && (
                        <tr>
                          <td colSpan={4} className="px-3 py-6 text-center text-slate-500">
                            ไม่มีข้อมูล
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setSelectedMonth(null)}
                    disabled={!selectedMonth}
                  >
                    เคลียร์การเลือก
                  </Button>
                  {selectedMonth && (
                    <div className="text-xs text-slate-400">
                      เดือนที่เลือก: <span className="font-medium text-slate-200">{selectedMonth}</span> ({selectedMonthSymbolData.length} Symbols)
                    </div>
                  )}
                </div>
              </div>

              {/* Column Chart: Symbol breakdown for selected month */}
              <div className="2xl:col-span-3 flex flex-col gap-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-slate-200">
                    กราฟมูลค่ารวม (USD) ราย Symbol {selectedMonth ? `เดือน ${selectedMonth}` : ""}
                  </h3>
                  <div className="text-xs text-slate-400">
                    {selectedMonth ? `${selectedMonthSymbolData.length} symbols` : "ยังไม่เลือกเดือน"}
                  </div>
                </div>
                <div className="w-full border border-slate-600 rounded-lg bg-slate-800 h-[500px] p-2">
                  {selectedMonth && selectedMonthSymbolData.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={selectedMonthSymbolData.map(d => ({
                          symbol: d.symbol,
                          net: d.net_amount_usd
                        })).reverse()}
                        margin={{ top: 10, right: 20, left: 10, bottom: 80 }}
                      >
                        <CartesianGrid vertical={false} strokeDasharray="3 3" stroke={colors.gridStroke} />
                        <XAxis
                          dataKey="symbol"
                          interval={0}
                          angle={-60}
                          textAnchor="end"
                          height={80}
                          tick={{ fontSize: 11, fill: colors.axisText }}
                        />
                        <YAxis tick={{ fontSize: 12, fill: colors.axisText }} />
                        <Tooltip
                          contentStyle={tooltipStyle as any}
                          formatter={(v:number)=>v.toLocaleString(undefined,{maximumFractionDigits:2})}
                          labelFormatter={(l)=>`Symbol: ${l}`}
                        />
                        <Bar dataKey="net" fill={colors.primary} />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-full flex items-center justify-center text-sm text-slate-500">
                      {selectedMonth
                        ? "ไม่มี Symbol ในเดือนที่เลือก (อาจถูกกรองออก)"
                        : "คลิกเลือกเดือนจากตารางสรุปรายเดือนทางซ้าย"}
                    </div>
                  )}
                </div>
                <div className="text-[11px] text-slate-500">
                  กราฟนี้แสดง Net Amount รวมต่อ Symbol สำหรับเดือนที่เลือก (ไม่รวม Option)
                </div>
              </div>
            </div>

            <Separator />

            {/* Month+Symbol Table */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold text-slate-200">ตาราง Month + Symbol (ไม่รวม Option)</h3>
                <div className="text-sm text-slate-400">
                  {monthlyBySymbol.length} แถว
                </div>
              </div>
              <div className="overflow-auto rounded-lg border border-slate-600 max-h-[500px]">
                <table className="w-full text-sm">
                  <thead className="bg-slate-700 text-slate-200">
                    <tr>
                      <th className="text-left px-3 py-2 font-medium">เดือน</th>
                      <th className="text-left px-3 py-2 font-medium">Symbol</th>
                      <th className="text-right px-3 py-2 font-medium">Orders</th>
                      <th className="text-right px-3 py-2 font-medium">Net (USD)</th>
                      <th className="text-right px-3 py-2 font-medium">Avg / Order</th>
                    </tr>
                  </thead>
                  <tbody>
                    {monthlyBySymbol.map(m => (
                      <tr key={m.month + "_" + m.symbol} className="border-t border-slate-700 hover:bg-slate-700/60">
                        <td className="px-3 py-2 font-medium whitespace-nowrap text-slate-100">{m.month}</td>
                        <td className="px-3 py-2 truncate max-w-[240px] text-slate-200" title={m.symbol}>{m.symbol}</td>
                        <td className="px-3 py-2 text-right text-slate-300">{m.orders.toLocaleString()}</td>
                        <td className="px-3 py-2 text-right text-slate-200">
                          {m.net_amount_usd.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                        </td>
                        <td className="px-3 py-2 text-right text-slate-300">
                          {m.avg_net_per_order_usd.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                        </td>
                      </tr>
                    ))}
                    {!monthlyBySymbol.length && (
                      <tr>
                        <td colSpan={5} className="px-3 py-6 text-center text-slate-500">
                          ยังไม่มีข้อมูล
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

          </CardContent>
        </Card>

        <div className="text-xs text-slate-500 space-y-1">
          <div>Dark Mode ตั้งเป็นค่าเริ่มต้น (Default) พร้อมปรับสีให้มี Contrast ที่อ่านง่าย</div>
          <div>รักษา logic การคำนวณ, โครงสร้างตาราง, การกรอง และ Option Exclusion ตามเดิม</div>
          <div>ปรับเฉพาะชั้น Presentation (UI theme) และเพิ่ม design tokens เพื่อรองรับการ maintain ในอนาคต</div>
        </div>

      </div>
    </div>
  );
};

export default Dashboard;
