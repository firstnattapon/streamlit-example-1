import React, { useState, useMemo, useEffect, useRef } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceDot,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';

type PointCompare = {
  x1: number;
  y1_delta1: number | null;
  y1_delta2: number | null;
  y2_delta1: number | null;
  y2_delta2: number | null;
  y4_piece: number | null;
  y5_piece: number | null;
  y3_delta1: number | null;
  y3_delta2: number | null;
  // Benchmarks
  y6_ref_delta1: number | null;
  y6_ref_delta2: number | null;
  y7_ref_delta1: number | null;
  y7_ref_delta2: number | null;
  // Intrinsic
  y8_call_intrinsic: number;
  y9_put_intrinsic: number;
};

const H = (z: number) => (z >= 0 ? 1 : 0);
const safeLog = (arg: number): number | null => (arg > 0 ? Math.log(arg) : null);
const piecewiseDelta = (x: number, thr: number, below: number, above: number) =>
  below + H(x - thr) * (above - below);
const scaleOrNull = (v: number | null, s: number): number | null => (v === null ? null : v * s);
const addBiasOrNull = (v: number | null, b: number): number | null => (v === null ? null : v + b);
const sumOrNull = (values: Array<number | null>, actives: boolean[]): number | null => {
  for (let i = 0; i < values.length; i++) {
    if (actives[i] && values[i] === null) return null;
  }
  return values.reduce((acc, v, i) => (actives[i] ? acc + (v as number) : acc), 0);
};

const LogarithmicGraph = () => {
  // --- Parameters / States (เดิม + bias) ---
  const [x0_1, setX0_1] = useState(7);
  const [x0_2, setX0_2] = useState(10);
  const [x1Range, setX1Range] = useState<[number, number]>([3.0, 17.0]);

  const [delta1, setDelta1] = useState(0.2);
  const [delta2, setDelta2] = useState(1.0);
  const [constant1, setConstant1] = useState(1500);
  const [constant2, setConstant2] = useState(1500);

  const [refConst, setRefConst] = useState(1500);
  const [b1, setB1] = useState(0);
  const [b2, setB2] = useState(0);

  const [showY1, setShowY1] = useState(true);
  const [showY2, setShowY2] = useState(false);
  const [showY3, setShowY3] = useState(false);
  const [showY4, setShowY4] = useState(false);
  const [showY5, setShowY5] = useState(false);

  const [showY6, setShowY6] = useState(true);
  const [showY7, setShowY7] = useState(true);

  // --- NEW: Intrinsic toggles & params ---
  const [showY8, setShowY8] = useState(true); // Call Intrinsic
  const [showY9, setShowY9] = useState(true); // Put Intrinsic
  const [callContracts, setCallContracts] = useState<number>(100);
  const [putContracts, setPutContracts] = useState<number>(100);
  const [premiumCall, setPremiumCall] = useState<number>(2.6); // per share/contract
  const [premiumPut, setPremiumPut] = useState<number>(2.6); // per share/contract
  const [includePremium, setIncludePremium] = useState<boolean>(true);

  // --- NEW: Auto roll-over β memory (y1,y5) ---
  const [autoRolloverB1, setAutoRolloverB1] = useState<boolean>(false);
  const [anchorPstar, setAnchorPstar] = useState<number>(13.79); // ราคาปัจจุบัน/จุดยึด (ตัวอย่างของคุณ)
  const b1BaseRef = useRef<number>(0); // จำฐาน b1 ตอนเปิด Auto

  // เมื่อสวิตช์เปลี่ยนสถานะ: เก็บฐานไว้เพื่อทำให้ idempotent
  useEffect(() => {
    if (autoRolloverB1) {
      b1BaseRef.current = b1;
    }
    // ปิด auto ไม่ทำอะไรกับ b1 (คงตามค่าปัจจุบัน)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRolloverB1]);

  // คำนวณ roll-over อัตโนมัติเมื่อค่าที่เกี่ยวข้องเปลี่ยน
  useEffect(() => {
    if (!autoRolloverB1) return;
    if (anchorPstar <= 0 || x0_1 <= 0) return; // กัน ln พัง
    const lnTerm = Math.log(anchorPstar / x0_1);
    // สูตร: b' = b_base + (refConst - constant1) * ln(P*/x0_1)
    const newB1 = b1BaseRef.current + (refConst - constant1) * lnTerm;
    if (Number.isFinite(newB1)) setB1(newB1);
  }, [autoRolloverB1, refConst, constant1, x0_1, anchorPstar]);

  const calcXDomain = (minX: number, maxX: number, ...xs: number[]) =>
    [Math.min(minX, ...xs), Math.max(maxX, ...xs)] as [number, number];

  const generateComparisonData = (): PointCompare[] => {
    const pts: PointCompare[] = [];
    const steps = 100;
    const step = (x1Range[1] - x1Range[0]) / steps;

    for (let i = 0; i <= steps; i++) {
      const x1 = x1Range[0] + i * step;

      const ln1 = safeLog(x1 / x0_1);
      const ln2 = safeLog(2 - x1 / x0_2);

      const y1_raw = ln1 === null ? null : constant1 * ln1;
      const y2_raw = ln2 === null ? null : constant2 * ln2;

      const y1_d1 = addBiasOrNull(scaleOrNull(y1_raw, delta1), b1);
      const y1_d2 = addBiasOrNull(scaleOrNull(y1_raw, delta2), b1);

      const y2_d1 = addBiasOrNull(scaleOrNull(y2_raw, delta1), b2);
      const y2_d2 = addBiasOrNull(scaleOrNull(y2_raw, delta2), b2);

      const d_y4 = piecewiseDelta(x1, x0_2, delta2, delta1);
      const y4_piece = addBiasOrNull(scaleOrNull(y2_raw, d_y4), b2);

      const d_y5 = piecewiseDelta(x1, x0_1, delta1, delta2);
      const y5_piece = addBiasOrNull(scaleOrNull(y1_raw, d_y5), b1);

      const y6_raw = ln1 === null ? null : refConst * ln1;
      const y7_raw = ln2 === null ? null : refConst * ln2;
      const y6_ref_d1 = scaleOrNull(y6_raw, delta1);
      const y6_ref_d2 = scaleOrNull(y6_raw, delta2);
      const y7_ref_d1 = scaleOrNull(y7_raw, delta1);
      const y7_ref_d2 = scaleOrNull(y7_raw, delta2);

      // Intrinsic (รวม premium แบบคงที่ต่อสัญญา)
      const premCallCost = includePremium ? callContracts * premiumCall : 0;
      const premPutCost = includePremium ? putContracts * premiumPut : 0;

      const y8_call_intrinsic = Math.max(0, x1 - x0_1) * callContracts - premCallCost;
      const y9_put_intrinsic = Math.max(0, x0_2 - x1) * putContracts - premPutCost;

      const actives_d1 = [showY1, showY2, showY4, showY5, showY8, showY9];
      const vals_d1 = [y1_d1, y2_d1, y4_piece, y5_piece, y8_call_intrinsic, y9_put_intrinsic] as (number | null)[];
      const y3_d1 = sumOrNull(vals_d1, actives_d1);

      const actives_d2 = [showY1, showY2, showY4, showY5, showY8, showY9];
      const vals_d2 = [y1_d2, y2_d2, y4_piece, y5_piece, y8_call_intrinsic, y9_put_intrinsic] as (number | null)[];
      const y3_d2 = sumOrNull(vals_d2, actives_d2);

      pts.push({
        x1,
        y1_delta1: y1_d1,
        y1_delta2: y1_d2,
        y2_delta1: y2_d1,
        y2_delta2: y2_d2,
        y4_piece,
        y5_piece,
        y3_delta1: y3_d1,
        y3_delta2: y3_d2,
        y6_ref_delta1: y6_ref_d1,
        y6_ref_delta2: y6_ref_d2,
        y7_ref_delta1: y7_ref_d1,
        y7_ref_delta2: y7_ref_d2,
        y8_call_intrinsic,
        y9_put_intrinsic,
      });
    }
    return pts;
  };

  const comparisonData = useMemo(
    () => generateComparisonData(),
    [
      x0_1, x0_2, x1Range,
      delta1, delta2,
      constant1, constant2,
      refConst,
      b1, b2,
      showY1, showY2, showY4, showY5,
      showY8, showY9,
      callContracts, putContracts,
      premiumCall, premiumPut,
      includePremium,
    ]
  );

  const dataDelta1 = useMemo(
    () =>
      comparisonData.map(d => ({
        x1: d.x1,
        y1: d.y1_delta1,
        y2: d.y2_delta1,
        y3: d.y3_delta1,
        y4: d.y4_piece,
        y5: d.y5_piece,
        y6: d.y6_ref_delta1,
        y7: d.y7_ref_delta1,
        y8: d.y8_call_intrinsic,
        y9: d.y9_put_intrinsic,
      })),
    [comparisonData]
  );

  const dataDelta2 = useMemo(
    () =>
      comparisonData.map(d => ({
        x1: d.x1,
        y1: d.y1_delta2,
        y2: d.y2_delta2,
        y3: d.y3_delta2,
        y4: d.y4_piece,
        y5: d.y5_piece,
        y6: d.y6_ref_delta2,
        y7: d.y7_ref_delta2,
        y8: d.y8_call_intrinsic,
        y9: d.y9_put_intrinsic,
      })),
    [comparisonData]
  );

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-gray-800 border border-gray-600 p-3 rounded-lg shadow-lg">
          <p className="text-cyan-400 font-semibold mb-2">{`x₁: ${Number(label).toFixed(2)}`}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }} className="text-sm">
              {`${entry.name}: ${
                entry.value !== null && entry.value !== undefined ? Math.round(Number(entry.value)) : 'ไม่กำหนด'
              }`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  const xDomain = useMemo(
    () => calcXDomain(x1Range[0], x1Range[1], x0_1, x0_2),
    [x1Range, x0_1, x0_2]
  );
  const yTickFormatter = (v: number) => Math.round(v).toString();

  const X_MIN = 0.1;
  const X_MAX = 50;
  const X_STEP = 0.1;
  const updateXMin = (val: number) => {
    const newMin = Math.min(Math.max(val, X_MIN), x1Range[1] - X_STEP);
    setX1Range([newMin, x1Range[1]]);
  };
  const updateXMax = (val: number) => {
    const newMax = Math.max(Math.min(val, X_MAX), x1Range[0] + X_STEP);
    setX1Range([x1Range[0], newMax]);
  };
  const resetXRange = () => setX1Range([3.0, 17.0]);

  const setBiasZero = () => {
    setB1(0);
    setB2(0);
    // อัพเดตฐานใหม่ถ้าเปิด Auto (เพื่อให้ฐานสอดคล้องค่าล่าสุด)
    if (autoRolloverB1) b1BaseRef.current = 0;
  };
  const setBiasDemo = () => {
    setB1(-1000);
    setB2(1000);
    if (autoRolloverB1) b1BaseRef.current = -1000;
  };

  // Break-even จุดอ้างอิงเมื่อรวม premium
  const beCall = useMemo(() => x0_1 + (includePremium ? premiumCall : 0), [x0_1, premiumCall, includePremium]);
  const bePut = useMemo(() => x0_2 - (includePremium ? premiumPut : 0), [x0_2, premiumPut, includePremium]);

  return (
    <div className="w-full h-full p-6 bg-gradient-to-br from-gray-900 to-slate-900">
      <Card className="w-full max-w-7xl mx-auto shadow-2xl bg-gray-800 border-gray-700">
        <CardHeader className="border-b border-gray-700">
          <CardTitle className="text-2xl font-bold text-center text-gray-100">
            เปรียบเทียบกราฟ (รองรับ β): + Intrinsic พร้อม premium toggle
          </CardTitle>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* แถบสมการ */}
          <div className="grid grid-cols-1 md:grid-cols-9 gap-4 p-4 bg-gray-900 rounded-lg border-2 border-gray-700 items-stretch auto-rows-fr">
            {/* y1 */}
            <div
              className={`text-center p-3 rounded transition-all flex flex-col h-full min-h-[190px] ${
                showY1 ? 'bg-cyan-900/40 border-2 border-cyan-500' : 'bg-gray-800 border-2 border-gray-600 opacity-50'
              }`}
            >
              <p className="font-semibold text-cyan-400">สมการที่ 1:</p>
              <p className="text-lg mt-2 text-gray-200">
                y₁ = <b>{Math.round(b1)}</b> + {constant1} × ln(x₁/x₀₁) × δ
              </p>
              <p className="text-sm text-cyan-400 mt-1">x₀₁ = {x0_1.toFixed(2)}</p>
              <div className="mt-auto pt-3 flex justify-center">
                <Button
                  size="sm"
                  variant={showY1 ? 'default' : 'outline'}
                  onClick={() => setShowY1(!showY1)}
                  className={showY1 ? 'bg-cyan-600 hover:bg-cyan-700 text-white' : 'border-gray-600 text-gray-400 hover:bg-gray-700'}
                >
                  {showY1 ? 'Active' : 'Inactive'}
                </Button>
              </div>
            </div>

            {/* y2 */}
            <div
              className={`text-center p-3 rounded transition-all flex flex-col h-full min-h-[190px] ${
                showY2 ? 'bg-amber-900/40 border-2 border-amber-500' : 'bg-gray-800 border-2 border-gray-600 opacity-50'
              }`}
            >
              <p className="font-semibold text-amber-400">สมการที่ 2 (เดิม):</p>
              <p className="text-lg mt-2 text-gray-200">
                y₂ = <b>{Math.round(b2)}</b> + {constant2} × ln(2 - x₁/x₀₂) × δ
              </p>
              <p className="text-sm text-amber-400 mt-1">x₀₂ = {x0_2.toFixed(2)}</p>
              <div className="mt-auto pt-3 flex justify-center">
                <Button
                  size="sm"
                  variant={showY2 ? 'default' : 'outline'}
                  onClick={() => setShowY2(!showY2)}
                  className={showY2 ? 'bg-amber-600 hover:bg-amber-700 text-white' : 'border-gray-600 text-gray-400 hover:bg-gray-700'}
                >
                  {showY2 ? 'Active' : 'Inactive'}
                </Button>
              </div>
            </div>

            {/* y3 */}
            <div
              className={`text-center p-3 rounded transition-all flex flex-col h-full min-h-[190px] ${
                showY3 ? 'bg-pink-900/40 border-2 border-pink-500' : 'bg-gray-800 border-2 border-gray-600 opacity-50'
              }`}
            >
              <p className="font-semibold text-pink-400">สมการที่ 3 (Net):</p>
              <p className="text-lg mt-2 text-gray-200">y₃ = Σ(เส้นที่ Active)</p>
              <p className="text-xs text-pink-400">* Benchmarks (y₆,y₇) ไม่ถูกรวมใน Net</p>
              <div className="mt-auto pt-3 flex justify-center">
                <Button
                  size="sm"
                  variant={showY3 ? 'default' : 'outline'}
                  onClick={() => setShowY3(!showY3)}
                  className={showY3 ? 'bg-pink-600 hover:bg-pink-700 text-white' : 'border-gray-600 text-gray-400 hover:bg-gray-700'}
                >
                  {showY3 ? 'Active' : 'Inactive'}
                </Button>
              </div>
            </div>

            {/* y4 */}
            <div
              className={`text-center p-3 rounded transition-all flex flex-col h-full min-h-[190px] ${
                showY4 ? 'bg-lime-900/40 border-2 border-lime-500' : 'bg-gray-800 border-2 border-gray-600 opacity-50'
              }`}
            >
              <p className="font-semibold text-lime-400">สมการที่ 4 (Piecewise δ บน y₂):</p>
              <p className="text-lg mt-2 text-gray-200">
                y₄ = <b>{Math.round(b2)}</b> + {constant2} × ln(2 - x₁/x₀₂) × <span className="font-semibold">δ(x₁)</span>
              </p>
              <p className="text-xs text-lime-300 mt-1">δ(x₁) = {'{ x₁ ≥ x₀₂ → δ₁ ; else → δ₂ }'}</p>
              <p className="text-sm text-lime-400 mt-1">threshold = x₀₂</p>
              <div className="mt-auto pt-3 flex justify-center">
                <Button
                  size="sm"
                  variant={showY4 ? 'default' : 'outline'}
                  onClick={() => setShowY4(!showY4)}
                  className={showY4 ? 'bg-lime-600 hover:bg-lime-700 text-white' : 'border-gray-600 text-gray-400 hover:bg-gray-700'}
                >
                  {showY4 ? 'Active' : 'Inactive'}
                </Button>
              </div>
            </div>

            {/* y5 */}
            <div
              className={`text-center p-3 rounded transition-all flex flex-col h-full min-h-[190px] ${
                showY5 ? 'bg-emerald-900/40 border-2 border-emerald-500' : 'bg-gray-800 border-2 border-gray-600 opacity-50'
              }`}
            >
              <p className="font-semibold text-emerald-400">สมการที่ 5 (Piecewise δ บน y₁):</p>
              <p className="text-lg mt-2 text-gray-200">
                y₅ = <b>{Math.round(b1)}</b> + {constant1} × ln(x₁/x₀₁) × <span className="font-semibold">δ(x₁)</span>
              </p>
              <p className="text-xs text-emerald-300 mt-1">δ(x₁) = {'{ x₁ ≥ x₀₁ → δ₂ ; else → δ₁ }'}</p>
              <p className="text-sm text-emerald-400 mt-1">threshold = x₀₁</p>
              <div className="mt-auto pt-3 flex justify-center">
                <Button
                  size="sm"
                  variant={showY5 ? 'default' : 'outline'}
                  onClick={() => setShowY5(!showY5)}
                  className={showY5 ? 'bg-emerald-600 hover:bg-emerald-700 text-white' : 'border-gray-600 text-gray-400 hover:bg-gray-700'}
                >
                  {showY5 ? 'Active' : 'Inactive'}
                </Button>
              </div>
            </div>

            {/* y6 */}
            <div
              className={`text-center p-3 rounded transition-all flex flex-col h-full min-h-[190px] ${
                showY6 ? 'bg-slate-800/40 border-2 border-slate-500' : 'bg-gray-800 border-2 border-gray-600 opacity-50'
              }`}
            >
              <p className="font-semibold text-slate-300">สมการที่ 6 (Benchmark y₁):</p>
              <p className="text-sm text-slate-300">y₆ = <b>0</b> + {refConst} × ln(x₁/x₀₁) × δ</p>
              <div className="mt-auto pt-3 flex justify-center">
                <Button
                  size="sm"
                  variant={showY6 ? 'default' : 'outline'}
                  onClick={() => setShowY6(!showY6)}
                  className={showY6 ? 'bg-slate-500 hover:bg-slate-600 text-white' : 'border-gray-600 text-gray-400 hover:bg-gray-700'}
                >
                  {showY6 ? 'Active' : 'Inactive'}
                </Button>
              </div>
            </div>

            {/* y7 */}
            <div
              className={`text-center p-3 rounded transition-all flex flex-col h-full min-h-[190px] ${
                showY7 ? 'bg-violet-900/30 border-2 border-violet-500' : 'bg-gray-800 border-2 border-gray-600 opacity-50'
              }`}
            >
              <p className="font-semibold text-violet-300">สมการที่ 7 (Benchmark y₂):</p>
              <p className="text-sm text-violet-200">y₇ = <b>0</b> + {refConst} × ln(2 − x₁/x₀₂) × δ</p>
              <div className="mt-auto pt-3 flex justify-center">
                <Button
                  size="sm"
                  variant={showY7 ? 'default' : 'outline'}
                  onClick={() => setShowY7(!showY7)}
                  className={showY7 ? 'bg-violet-500 hover:bg-violet-600 text-white' : 'border-gray-600 text-gray-400 hover:bg-gray-700'}
                >
                  {showY7 ? 'Active' : 'Inactive'}
                </Button>
              </div>
            </div>

            {/* y8 */}
            <div
              className={`text-center p-3 rounded transition-all flex flex-col h-full min-h-[190px] ${
                showY8 ? 'bg-red-900/40 border-2 border-red-500' : 'bg-gray-800 border-2 border-gray-600 opacity-50'
              }`}
            >
              <p className="font-semibold text-red-400">สมการที่ 8 (Call Intrinsic):</p>
              <p className="text-sm text-red-300">
                y₈ = max(0, x₁ − x₀₁) × contracts_call {includePremium ? '− contracts_call × premium_call' : ''}
              </p>
              <div className="mt-auto pt-3 flex justify-center">
                <Button
                  size="sm"
                  variant={showY8 ? 'default' : 'outline'}
                  onClick={() => setShowY8(!showY8)}
                  className={showY8 ? 'bg-red-600 hover:bg-red-700 text-white' : 'border-gray-600 text-gray-400 hover:bg-gray-700'}
                >
                  {showY8 ? 'Active' : 'Inactive'}
                </Button>
              </div>
            </div>

            {/* y9 */}
            <div
              className={`text-center p-3 rounded transition-all flex flex-col h-full min-h-[190px] ${
                showY9 ? 'bg-green-900/40 border-2 border-green-500' : 'bg-gray-800 border-2 border-gray-600 opacity-50'
              }`}
            >
              <p className="font-semibold text-green-400">สมการที่ 9 (Put Intrinsic):</p>
              <p className="text-sm text-green-300">
                y₉ = max(0, x₀₂ − x₁) × contracts_put {includePremium ? '− contracts_put × premium_put' : ''}
              </p>
              <div className="mt-auto pt-3 flex justify-center">
                <Button
                  size="sm"
                  variant={showY9 ? 'default' : 'outline'}
                  onClick={() => setShowY9(!showY9)}
                  className={showY9 ? 'bg-green-600 hover:bg-green-700 text-white' : 'border-gray-600 text-gray-400 hover:bg-gray-700'}
                >
                  {showY9 ? 'Active' : 'Inactive'}
                </Button>
              </div>
            </div>
          </div>

          {/* ปุ่มควบคุมรวม */}
          <div className="flex flex-wrap justify-center gap-3 p-3 bg-gray-900 rounded-lg border border-gray-700">
            <Button
              onClick={() => {
                setShowY1(true);
                setShowY2(true);
                setShowY3(true);
                setShowY4(true);
                setShowY5(true);
                setShowY6(true);
                setShowY7(true);
                setShowY8(true);
                setShowY9(true);
              }}
              className="bg-green-600 hover:bg-green-700 text-white"
            >
              เปิดทั้งหมด
            </Button>
            <Button
              onClick={() => {
                setShowY1(false);
                setShowY2(false);
                setShowY3(false);
                setShowY4(false);
                setShowY5(false);
                setShowY6(false);
                setShowY7(false);
                setShowY8(false);
                setShowY9(false);
              }}
              className="border-gray-600 text-gray-300 hover:bg-gray-700"
              variant="outline"
            >
              ปิดทั้งหมด
            </Button>
            <Button
              onClick={() => {
                setShowY1(false);
                setShowY2(false);
                setShowY3(true);
                setShowY4(false);
                setShowY5(false);
                setShowY6(false);
                setShowY7(false);
              }}
              className="bg-pink-600 hover:bg-pink-700 text-white"
            >
              Net เท่านั้น
            </Button>
            <Button onClick={setBiasZero} className="bg-slate-700 hover:bg-slate-600 text-white">
              รีเซ็ต β = 0
            </Button>
            <Button onClick={setBiasDemo} className="bg-amber-600 hover:bg-amber-500 text-white">
              เดโม β: b₁=-1000, b₂=+1000
            </Button>
          </div>

          {/* Tabs */}
          <Tabs defaultValue="comparison" className="w-full">
            <TabsList className="grid w-full grid-cols-4 bg-gray-900 border border-gray-700">
              <TabsTrigger value="comparison" className="data-[state=active]:bg-gray-700 data-[state=active]:text-white text-gray-400">
                เปรียบเทียบทั้งหมด
              </TabsTrigger>
              <TabsTrigger value="net" className="data-[state=active]:bg-gray-700 data-[state=active]:text-white text-gray-400">
                Net เท่านั้น
              </TabsTrigger>
              <TabsTrigger value="delta1" className="data-[state=active]:bg-gray-700 data-[state=active]:text-white text-gray-400">
                δ = {delta1.toFixed(2)}
              </TabsTrigger>
              <TabsTrigger value="delta2" className="data-[state=active]:bg-gray-700 data-[state=active]:text-white text-gray-400">
                δ = {delta2.toFixed(2)}
              </TabsTrigger>
            </TabsList>

            {/* เปรียบเทียบทั้งหมด */}
            <TabsContent value="comparison" className="space-y-4">
              <div className="bg-gray-900 p-4 rounded-lg border border-gray-700">
                <h3 className="text-lg font-semibold mb-3 text-center text-gray-100">
                  ครบชุด: y₁,y₂,y₄,y₅, Net, Benchmarks, y₈(call), y₉(put) + BE จุดคุ้มทุนเมื่อรวม premium
                </h3>
                <ResponsiveContainer width="100%" height={470}>
                  <LineChart data={comparisonData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis
                      type="number"
                      dataKey="x1"
                      domain={[xDomain[0], xDomain[1]]}
                      allowDataOverflow
                      label={{ value: 'x₁', position: 'insideBottom', offset: -5, fill: '#9ca3af' }}
                      stroke="#9ca3af"
                      tick={{ fill: '#9ca3af' }}
                    />
                    <YAxis
                      domain={[(min: number) => Math.min(min, 0), (max: number) => Math.max(max, 0)]}
                      label={{ value: 'y', angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
                      stroke="#9ca3af"
                      tick={{ fill: '#9ca3af' }}
                      tickFormatter={yTickFormatter}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend wrapperStyle={{ color: '#e5e7eb' }} />

                    {/* y1 */}
                    {showY1 && (
                      <Line
                        type="monotone"
                        dataKey="y1_delta1"
                        stroke="#06b6d4"
                        strokeWidth={2}
                        name={`y₁ (δ=${delta1.toFixed(2)})`}
                        dot={false}
                        strokeDasharray="5 5"
                      />
                    )}
                    {showY1 && (
                      <Line
                        type="monotone"
                        dataKey="y1_delta2"
                        stroke="#22d3ee"
                        strokeWidth={3}
                        name={`y₁ (δ=${delta2.toFixed(2)})`}
                        dot={false}
                      />
                    )}

                    {/* y2 */}
                    {showY2 && (
                      <Line
                        type="monotone"
                        dataKey="y2_delta1"
                        stroke="#fbbf24"
                        strokeWidth={2}
                        name={`y₂ (δ=${delta1.toFixed(2)})`}
                        dot={false}
                        strokeDasharray="5 5"
                      />
                    )}
                    {showY2 && (
                      <Line
                        type="monotone"
                        dataKey="y2_delta2"
                        stroke="#fde047"
                        strokeWidth={3}
                        name={`y₂ (δ=${delta2.toFixed(2)})`}
                        dot={false}
                      />
                    )}

                    {/* y4, y5 */}
                    {showY4 && (
                      <Line type="monotone" dataKey="y4_piece" stroke="#a3e635" strokeWidth={3} name="y₄ (piecewise δ, x₀₂, +b₂)" dot={false} />
                    )}
                    {showY5 && (
                      <Line type="monotone" dataKey="y5_piece" stroke="#10b981" strokeWidth={3} name="y₅ (piecewise δ, x₀₁ — δ สลับ, +b₁)" dot={false} />
                    )}

                    {/* Net */}
                    {showY3 && (
                      <Line
                        type="monotone"
                        dataKey="y3_delta1"
                        stroke="#ec4899"
                        strokeWidth={2.5}
                        name="Net (δ₁ base)"
                        dot={false}
                        strokeDasharray="5 5"
                      />
                    )}
                    {showY3 && (
                      <Line type="monotone" dataKey="y3_delta2" stroke="#f472b6" strokeWidth={3.5} name="Net (δ₂ base)" dot={false} />
                    )}

                    {/* Benchmarks */}
                    {showY6 && (
                      <Line
                        type="monotone"
                        dataKey="y6_ref_delta2"
                        stroke="#94a3b8"
                        strokeWidth={2.5}
                        name="y₆ (Ref y₁, δ₂)"
                        dot={false}
                        strokeDasharray="6 4"
                      />
                    )}
                    {showY7 && (
                      <Line
                        type="monotone"
                        dataKey="y7_ref_delta2"
                        stroke="#c084fc"
                        strokeWidth={2.5}
                        name="y₇ (Ref y₂, δ₂)"
                        dot={false}
                        strokeDasharray="6 4"
                      />
                    )}

                    {/* Intrinsic */}
                    {showY8 && <Line type="monotone" dataKey="y8_call_intrinsic" stroke="#ef4444" strokeWidth={3} name="y₈ (Call Intrinsic)" dot={false} />}
                    {showY9 && <Line type="monotone" dataKey="y9_put_intrinsic" stroke="#22c55e" strokeWidth={3} name="y₉ (Put Intrinsic)" dot={false} />}

                    {/* Reference Dots x0 */}
                    {(showY1 || showY5 || showY6 || showY8) && (
                      <ReferenceDot
                        x={x0_1}
                        y={0}
                        r={6}
                        fill="rgba(2,132,199,0.25)"
                        stroke="#06b6d4"
                        strokeWidth={2}
                        label={{ value: 'x₀₁', position: 'top', fill: '#9ca3af' }}
                      />
                    )}
                    {(showY2 || showY4 || showY7 || showY9) && (
                      <ReferenceDot
                        x={x0_2}
                        y={0}
                        r={6}
                        fill="rgba(217,119,6,0.25)"
                        stroke="#fbbf24"
                        strokeWidth={2}
                        label={{ value: 'x₀₂', position: 'top', fill: '#9ca3af' }}
                      />
                    )}

                    {/* Break-even marks */}
                    {includePremium && showY8 && (
                      <ReferenceDot
                        x={beCall}
                        y={0}
                        r={5}
                        fill="rgba(239,68,68,0.15)"
                        stroke="#ef4444"
                        strokeWidth={2}
                        label={{ value: 'BE₈', position: 'top', fill: '#ef4444' }}
                      />
                    )}
                    {includePremium && showY9 && (
                      <ReferenceDot
                        x={bePut}
                        y={0}
                        r={5}
                        fill="rgba(34,197,94,0.15)"
                        stroke="#22c55e"
                        strokeWidth={2}
                        label={{ value: 'BE₉', position: 'top', fill: '#22c55e' }}
                      />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>

            {/* Net เท่านั้น */}
            <TabsContent value="net" className="space-y-4">
              <div className="bg-gray-900 p-4 rounded-lg border border-gray-700">
                <h3 className="text-lg font-semibold mb-3 text-center text-gray-100">
                  เปรียบเทียบกราฟ Net (y₃) + Benchmark — δ₁ base vs δ₂ base
                </h3>
                <ResponsiveContainer width="100%" height={450}>
                  <LineChart data={comparisonData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis
                      type="number"
                      dataKey="x1"
                      domain={[xDomain[0], xDomain[1]]}
                      allowDataOverflow
                      label={{ value: 'x₁', position: 'insideBottom', offset: -5, fill: '#9ca3af' }}
                      stroke="#9ca3af"
                      tick={{ fill: '#9ca3af' }}
                    />
                    <YAxis
                      domain={[(min: number) => Math.min(min, 0), (max: number) => Math.max(max, 0)]}
                      label={{ value: 'y', angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
                      stroke="#9ca3af"
                      tick={{ fill: '#9ca3af' }}
                      tickFormatter={yTickFormatter}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend wrapperStyle={{ color: '#e5e7eb' }} />

                    {showY3 && <Line type="monotone" dataKey="y3_delta1" stroke="#ec4899" strokeWidth={3.5} name="Net (δ₁ base)" dot={false} />}
                    {showY3 && <Line type="monotone" dataKey="y3_delta2" stroke="#f472b6" strokeWidth={3.5} name="Net (δ₂ base)" dot={false} />}

                    {showY6 && (
                      <Line
                        type="monotone"
                        dataKey="y6_ref_delta2"
                        stroke="#94a3b8"
                        strokeWidth={3}
                        name="Benchmark (y₆, δ₂)"
                        dot={false}
                        strokeDasharray="6 4"
                      />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>

            {/* δ1 */}
            <TabsContent value="delta1" className="space-y-4">
              <div className="bg-gray-900 p-4 rounded-lg border border-gray-700">
                <h3 className="text-lg font-semibold mb-3 text-center text-gray-100">กราฟด้วย δ = {delta1.toFixed(2)}</h3>
                <ResponsiveContainer width="100%" height={450}>
                  <LineChart data={dataDelta1}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis
                      type="number"
                      dataKey="x1"
                      domain={[xDomain[0], xDomain[1]]}
                      allowDataOverflow
                      label={{ value: 'x₁', position: 'insideBottom', offset: -5, fill: '#9ca3af' }}
                      stroke="#9ca3af"
                      tick={{ fill: '#9ca3af' }}
                    />
                    <YAxis
                      domain={[(min: number) => Math.min(min, 0), (max: number) => Math.max(max, 0)]}
                      label={{ value: 'y', angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
                      stroke="#9ca3af"
                      tick={{ fill: '#9ca3af' }}
                      tickFormatter={yTickFormatter}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend wrapperStyle={{ color: '#e5e7eb' }} />

                    {/* y1..y7 */}
                    {showY1 && <Line type="monotone" dataKey="y1" stroke="#06b6d4" strokeWidth={3} name="y₁" dot={false} />}
                    {showY2 && <Line type="monotone" dataKey="y2" stroke="#fbbf24" strokeWidth={3} name="y₂ (เดิม, δ₁)" dot={false} />}
                    {showY4 && <Line type="monotone" dataKey="y4" stroke="#a3e635" strokeWidth={3} name="y₄ (piecewise δ, x₀₂, +b₂)" dot={false} />}
                    {showY5 && <Line type="monotone" dataKey="y5" stroke="#10b981" strokeWidth={3} name="y₅ (piecewise δ, x₀₁ — δ สลับ, +b₁)" dot={false} />}
                    {showY6 && <Line type="monotone" dataKey="y6" stroke="#94a3b8" strokeWidth={2.5} name="y₆ (Ref y₁, δ₁)" dot={false} strokeDasharray="6 4" />}
                    {showY7 && <Line type="monotone" dataKey="y7" stroke="#c084fc" strokeWidth={2.5} name="y₇ (Ref y₂, δ₁)" dot={false} strokeDasharray="6 4" />}
                    {showY3 && <Line type="monotone" dataKey="y3" stroke="#ec4899" strokeWidth={3.5} name="y₃ (Net)" dot={false} />}

                    {/* Intrinsic */}
                    {showY8 && <Line type="monotone" dataKey="y8" stroke="#ef4444" strokeWidth={3} name="y₈ (Call Intrinsic)" dot={false} />}
                    {showY9 && <Line type="monotone" dataKey="y9" stroke="#22c55e" strokeWidth={3} name="y₉ (Put Intrinsic)" dot={false} />}

                    {(showY1 || showY5 || showY6 || showY8) && (
                      <ReferenceDot
                        x={x0_1}
                        y={0}
                        r={6}
                        fill="rgba(2,132,199,0.25)"
                        stroke="#06b6d4"
                        strokeWidth={2}
                        label={{ value: 'x₀₁', position: 'top', fill: '#9ca3af' }}
                      />
                    )}
                    {(showY2 || showY4 || showY7 || showY9) && (
                      <ReferenceDot
                        x={x0_2}
                        y={0}
                        r={6}
                        fill="rgba(217,119,6,0.25)"
                        stroke="#fbbf24"
                        strokeWidth={2}
                        label={{ value: 'x₀₂', position: 'top', fill: '#9ca3af' }}
                      />
                    )}
                    {includePremium && showY8 && (
                      <ReferenceDot
                        x={beCall}
                        y={0}
                        r={5}
                        fill="rgba(239,68,68,0.15)"
                        stroke="#ef4444"
                        strokeWidth={2}
                        label={{ value: 'BE₈', position: 'top', fill: '#ef4444' }}
                      />
                    )}
                    {includePremium && showY9 && (
                      <ReferenceDot
                        x={bePut}
                        y={0}
                        r={5}
                        fill="rgba(34,197,94,0.15)"
                        stroke="#22c55e"
                        strokeWidth={2}
                        label={{ value: 'BE₉', position: 'top', fill: '#22c55e' }}
                      />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>

            {/* δ2 */}
            <TabsContent value="delta2" className="space-y-4">
              <div className="bg-gray-900 p-4 rounded-lg border border-gray-700">
                <h3 className="text-lg font-semibold mb-3 text-center text-gray-100">กราฟด้วย δ = {delta2.toFixed(2)}</h3>
                <ResponsiveContainer width="100%" height={450}>
                  <LineChart data={dataDelta2}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis
                      type="number"
                      dataKey="x1"
                      domain={[xDomain[0], xDomain[1]]}
                      allowDataOverflow
                      label={{ value: 'x₁', position: 'insideBottom', offset: -5, fill: '#9ca3af' }}
                      stroke="#9ca3af"
                      tick={{ fill: '#9ca3af' }}
                    />
                    <YAxis
                      domain={[(min: number) => Math.min(min, 0), (max: number) => Math.max(max, 0)]}
                      label={{ value: 'y', angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
                      stroke="#9ca3af"
                      tick={{ fill: '#9ca3af' }}
                      tickFormatter={yTickFormatter}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend wrapperStyle={{ color: '#e5e7eb' }} />

                    {showY1 && <Line type="monotone" dataKey="y1" stroke="#22d3ee" strokeWidth={3} name="y₁" dot={false} />}
                    {showY2 && <Line type="monotone" dataKey="y2" stroke="#fde047" strokeWidth={3} name="y₂ (เดิม, δ₂)" dot={false} />}
                    {showY4 && <Line type="monotone" dataKey="y4" stroke="#a3e635" strokeWidth={3} name="y₄ (piecewise δ, x₀₂, +b₂)" dot={false} />}
                    {showY5 && <Line type="monotone" dataKey="y5" stroke="#10b981" strokeWidth={3} name="y₅ (piecewise δ, x₀₁ — δ สลับ, +b₁)" dot={false} />}
                    {showY6 && <Line type="monotone" dataKey="y6" stroke="#94a3b8" strokeWidth={2.5} name="y₆ (Ref y₁, δ₂)" dot={false} strokeDasharray="6 4" />}
                    {showY7 && <Line type="monotone" dataKey="y7" stroke="#c084fc" strokeWidth={2.5} name="y₇ (Ref y₂, δ₂)" dot={false} strokeDasharray="6 4" />}
                    {showY3 && <Line type="monotone" dataKey="y3" stroke="#f472b6" strokeWidth={3.5} name="ย₃ (Net)" dot={false} />}

                    {showY8 && <Line type="monotone" dataKey="y8" stroke="#ef4444" strokeWidth={3} name="y₈ (Call Intrinsic)" dot={false} />}
                    {showY9 && <Line type="monotone" dataKey="y9" stroke="#22c55e" strokeWidth={3} name="y₉ (Put Intrinsic)" dot={false} />}

                    {(showY1 || showY5 || showY6 || showY8) && (
                      <ReferenceDot
                        x={x0_1}
                        y={0}
                        r={6}
                        fill="rgba(34,211,238,0.25)"
                        stroke="#22d3ee"
                        strokeWidth={2}
                        label={{ value: 'x₀₁', position: 'top', fill: '#9ca3af' }}
                      />
                    )}
                    {(showY2 || showY4 || showY7 || showY9) && (
                      <ReferenceDot
                        x={x0_2}
                        y={0}
                        r={6}
                        fill="rgba(253,224,71,0.25)"
                        stroke="#fde047"
                        strokeWidth={2}
                        label={{ value: 'x₀₂', position: 'top', fill: '#9ca3af' }}
                      />
                    )}
                    {includePremium && showY8 && (
                      <ReferenceDot
                        x={beCall}
                        y={0}
                        r={5}
                        fill="rgba(239,68,68,0.15)"
                        stroke="#ef4444"
                        strokeWidth={2}
                        label={{ value: 'BE₈', position: 'top', fill: '#ef4444' }}
                      />
                    )}
                    {includePremium && showY9 && (
                      <ReferenceDot
                        x={bePut}
                        y={0}
                        r={5}
                        fill="rgba(34,197,94,0.15)"
                        stroke="#22c55e"
                        strokeWidth={2}
                        label={{ value: 'BE₉', position: 'top', fill: '#22c55e' }}
                      />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>
          </Tabs>

          {/* Controls */}
          <div className="space-y-6 p-4 bg-gray-900 rounded-lg border border-gray-700">
            {/* ค่าคงที่ */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-3 p-3 bg-cyan-900/30 rounded-lg border-2 border-cyan-700">
                <div className="flex justify-between items-center">
                  <Label className="text-lg font-semibold text-cyan-400">ค่าคงที่สมการที่ 1:</Label>
                  <span className="text-2xl font-bold text-cyan-300">{constant1}</span>
                </div>
                <Slider value={[constant1]} onValueChange={(v) => setConstant1(v[0])} min={100} max={5000} step={1} />
                <p className="text-xs text-cyan-400 text-center">ช่วง: 100 - 5000</p>
              </div>

              <div className="space-y-3 p-3 bg-teal-900/30 rounded-lg border-2 border-teal-700">
                <div className="flex justify-between items-center">
                  <Label className="text-lg font-semibold text-teal-400">ค่าคงที่สมการที่ 2/4:</Label>
                  <span className="text-2xl font-bold text-teal-300">{constant2}</span>
                </div>
                <Slider value={[constant2]} onValueChange={(v) => setConstant2(v[0])} min={100} max={5000} step={1} />
                <p className="text-xs text-teal-400 text-center">ช่วง: 100 - 5000</p>
              </div>

              <div className="space-y-3 p-3 bg-slate-800/40 rounded-lg border-2 border-slate-600">
                <div className="flex justify-between items-center">
                  <Label className="text-lg font-semibold text-slate-200">ค่าคงที่ baseline (y₆/y₇):</Label>
                  <span className="text-2xl font-bold text-slate-100">{refConst}</span>
                </div>
                <Slider value={[refConst]} onValueChange={(v) => setRefConst(v[0])} min={100} max={5000} step={1} />
                <p className="text-xs text-slate-300 text-center">ช่วง: 100 - 5000</p>
              </div>
            </div>

            {/* Bias */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-3 p-3 bg-sky-900/30 rounded-lg border-2 border-sky-700">
                <div className="flex justify-between items-center">
                  <Label className="text-lg font-semibold text-sky-400">b₁ (bias ของ y₁ / y₅):</Label>
                  <span className="text-2xl font-bold text-sky-300">{Math.round(b1)}</span>
                </div>
                <Slider value={[b1]} onValueChange={(v) => setB1(v[0])} min={-5000} max={5000} step={1} />
                <p className="text-xs text-sky-400 text-center">ช่วง: −5000 ถึง +5000</p>
              </div>

              <div className="space-y-3 p-3 bg-amber-900/30 rounded-lg border-2 border-amber-700">
                <div className="flex justify-between items-center">
                  <Label className="text-lg font-semibold text-amber-300">b₂ (bias ของ y₂ / y₄):</Label>
                  <span className="text-2xl font-bold text-amber-200">{b2}</span>
                </div>
                <Slider value={[b2]} onValueChange={(v) => setB2(v[0])} min={-5000} max={5000} step={1} />
                <p className="text-xs text-amber-200 text-center">ช่วง: −5000 ถึง +5000</p>
              </div>
            </div>

            {/* X range */}
            <div className="space-y-3 p-3 bg-slate-900/30 rounded-lg border-2 border-slate-700">
              <div className="flex items-center justify-between">
                <Label className="text-lg font-semibold text-slate-200">ช่วงแกน x₁</Label>
                <div className="flex items-center gap-2">
                  <span className="text-sm text-slate-300">
                    [{x1Range[0].toFixed(2)} , {x1Range[1].toFixed(2)}]
                  </span>
                  <Button
                    size="sm"
                    variant="outline"
                    className="border-gray-600 text-gray-300 hover:bg-gray-700"
                    onClick={resetXRange}
                  >
                    รีเซ็ต
                  </Button>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-slate-300">x₁ min</span>
                    <span className="text-sm text-slate-300">{x1Range[0].toFixed(2)}</span>
                  </div>
                  <Slider value={[x1Range[0]]} onValueChange={(v) => updateXMin(v[0])} min={0.1} max={50 - 0.1} step={0.1} />
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-slate-300">x₁ max</span>
                    <span className="text-sm text-slate-300">{x1Range[1].toFixed(2)}</span>
                  </div>
                  <Slider value={[x1Range[1]]} onValueChange={(v) => updateXMax(v[0])} min={0.2} max={50} step={0.1} />
                </div>
              </div>
              <p className="text-xs text-slate-400 text-center">ช่วงที่อนุญาต: 0.1 ถึง 50 (step 0.1)</p>
            </div>

            {/* Deltas */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-3 p-3 bg-rose-900/30 rounded-lg border-2 border-rose-700">
                <div className="flex justify-between items-center">
                  <Label className="text-lg font-semibold text-rose-400">Delta 1 (δ₁):</Label>
                  <span className="text-2xl font-bold text-rose-300">{delta1.toFixed(2)}</span>
                </div>
                <Slider value={[delta1]} onValueChange={(v) => setDelta1(v[0])} min={0.01} max={2} step={0.01} />
                <p className="text-xs text-rose-400 text-center">ช่วง: 0.01 - 2.00</p>
              </div>

              <div className="space-y-3 p-3 bg-indigo-900/30 rounded-lg border-2 border-indigo-700">
                <div className="flex justify-between items-center">
                  <Label className="text-lg font-semibold text-indigo-400">Delta 2 (δ₂):</Label>
                  <span className="text-2xl font-bold text-indigo-300">{delta2.toFixed(2)}</span>
                </div>
                <Slider value={[delta2]} onValueChange={(v) => setDelta2(v[0])} min={0.01} max={2} step={0.01} />
                <p className="text-xs text-indigo-400 text-center">ช่วง: 0.01 - 2.00</p>
              </div>
            </div>

            {/* x0s */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-3 p-3 bg-blue-900/30 rounded-lg border-2 border-blue-700">
                <div className="flex justify-between items-center">
                  <Label className="text-lg font-semibold text-blue-400">ค่า x₀₁ (กราฟที่ 1 / threshold y₅):</Label>
                  <span className="text-xl font-bold text-blue-300">{x0_1.toFixed(2)}</span>
                </div>
                <Slider value={[x0_1]} onValueChange={(v) => setX0_1(v[0])} min={0.1} max={50} step={0.1} />
                <p className="text-xs text-blue-400 text-center">ช่วง: 0.1 - 50</p>
              </div>

              <div className="space-y-3 p-3 bg-green-900/30 rounded-lg border-2 border-green-700">
                <div className="flex justify-between items-center">
                  <Label className="text-lg font-semibold text-green-400">ค่า x₀₂ (กราฟที่ 2 / threshold y₄):</Label>
                  <span className="text-xl font-bold text-green-300">{x0_2.toFixed(2)}</span>
                </div>
                <Slider value={[x0_2]} onValueChange={(v) => setX0_2(v[0])} min={0.1} max={50} step={0.1} />
                <p className="text-xs text-green-400 text-center">ช่วง: 0.1 - 50</p>
              </div>
            </div>

            {/* NEW: Auto roll-over β memory (y1,y5) */}
            <div className="space-y-4 p-4 bg-zinc-900/40 rounded-lg border-2 border-zinc-700">
              <div className="flex items-center justify-between">
                <Label className="text-lg font-semibold text-zinc-200">Auto roll-over β (สำหรับ y₁ + y₅)</Label>
                <div className="flex items-center gap-3">
                  <Switch checked={autoRolloverB1} onCheckedChange={setAutoRolloverB1} />
                  <span className="text-sm text-zinc-300">{autoRolloverB1 ? 'On' : 'Off'}</span>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-sm font-medium text-zinc-200">
                    Anchor P* (จุดยึดสำหรับความต่อเนื่อง): {anchorPstar.toFixed(2)}
                  </Label>
                  <span className="text-xs text-zinc-400">ต้อง &gt; 0 และสอดคล้องโดเมน ln(P*/x₀₁)</span>
                </div>
                <Slider value={[anchorPstar]} onValueChange={(v) => setAnchorPstar(v[0])} min={0.1} max={50} step={0.01} />
              </div>

              <p className="text-xs text-zinc-300">
                สูตร: <span className="text-zinc-100">b₁ = b₁(base) + (refConst − constant1) × ln(P*/x₀₁)</span> &nbsp;|&nbsp; base จะถูกจดจำเมื่อกด ON
              </p>
            </div>

            {/* Intrinsic controls */}
            <div className="space-y-4 p-4 bg-zinc-900/40 rounded-lg border-2 border-zinc-700">
              <div className="flex items-center justify-between">
                <Label className="text-lg font-semibold text-zinc-200">Include premium in P/L</Label>
                <div className="flex items-center gap-3">
                  <Switch checked={includePremium} onCheckedChange={setIncludePremium} />
                  <span className="text-sm text-zinc-300">{includePremium ? 'On' : 'Off'}</span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-3 p-3 bg-red-900/30 rounded-lg border-2 border-red-700">
                  <div className="flex justify-between items-center">
                    <Label className="text-lg font-semibold text-red-300">contracts_call (y₈):</Label>
                    <span className="text-2xl font-bold text-red-200">{callContracts}</span>
                  </div>
                  <Slider value={[callContracts]} onValueChange={(v) => setCallContracts(v[0])} min={0} max={10000} step={1} />
                  <div className="flex justify-between items-center">
                    <Label className="text-sm font-medium text-red-200">premium_call</Label>
                    <span className="text-lg font-bold text-red-100">{premiumCall.toFixed(2)}</span>
                  </div>
                  <Slider value={[premiumCall]} onValueChange={(v) => setPremiumCall(v[0])} min={0} max={1000} step={0.01} />
                </div>

                <div className="space-y-3 p-3 bg-green-900/30 rounded-lg border-2 border-green-700">
                  <div className="flex justify-between items-center">
                    <Label className="text-lg font-semibold text-green-300">contracts_put (y₉):</Label>
                    <span className="text-2xl font-bold text-green-200">{putContracts}</span>
                  </div>
                  <Slider value={[putContracts]} onValueChange={(v) => setPutContracts(v[0])} min={0} max={10000} step={1} />
                  <div className="flex justify-between items-center">
                    <Label className="text-sm font-medium text-green-200">premium_put</Label>
                    <span className="text-lg font-bold text-green-100">{premiumPut.toFixed(2)}</span>
                  </div>
                  <Slider value={[premiumPut]} onValueChange={(v) => setPremiumPut(v[0])} min={0} max={1000} step={0.01} />
                </div>
              </div>

              <p className="text-xs text-zinc-300">
                * Break-even: BE₈ = x₀₁ + premium_call, BE₉ = x₀₂ − premium_put (แสดงเฉพาะเมื่อเปิด Include premium และเปิดเส้นนั้น ๆ)
              </p>
            </div>
          </div>

          <div className="p-4 bg-amber-900/30 rounded-lg border border-amber-700">
            <p className="text-sm text-amber-200">
              <strong className="text-amber-400">หมายเหตุ:</strong> β-memory ใช้ baseline = refConst และ pivot = x₀₁;
              ถ้าเปลี่ยน anchor ให้แน่ใจว่าโดเมน ln(P*/x₀₁) &gt; 0 เสมอ (ตัวอย่างคุณ: P* = 13.79, x₀₁ = 7 ⇒ ln &gt; 0).
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default LogarithmicGraph;
