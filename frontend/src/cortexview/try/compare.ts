/**
 * Everyday comparisons for the measured numbers, so a patient can picture a size.
 * Fixed reference sizes and fixed sentences; the only input is the number from the metrics engine.
 */

/** Typical widths (mm). Standard sizes where one exists (balls, coins). */
const WIDTHS: { mm: number; what: string }[] = [
  { mm: 10, what: 'a pea' },
  { mm: 19, what: 'a US penny' },
  { mm: 24, what: 'a US quarter' },
  { mm: 40, what: 'a ping-pong ball' },
  { mm: 43, what: 'a golf ball' },
  { mm: 67, what: 'a tennis ball' },
  { mm: 74, what: 'a baseball' },
]

export function widthComparison(mm: number): string {
  const best = WIDTHS.reduce((a, b) => (Math.abs(Math.log(b.mm / mm)) < Math.abs(Math.log(a.mm / mm)) ? b : a))
  const ratio = mm / best.mm
  const how = ratio < 0.85 ? 'a bit smaller than' : ratio > 1.15 ? 'a bit wider than' : 'about the width of'
  return `${how} ${best.what} (${best.mm} mm)`
}

/** Kitchen measures: a teaspoon is 5 mL, a tablespoon 15 mL. */
export function volumeComparison(ml: number): string {
  if (ml < 2.5) return 'less than half a teaspoon'
  if (ml < 12) {
    const tsp = Math.round(ml / 5)
    return tsp <= 1 ? 'about a teaspoon (5 mL)' : `about ${tsp} teaspoons (5 mL each)`
  }
  const tbsp = Math.round(ml / 15)
  return tbsp <= 1 ? 'about a tablespoon (15 mL)' : `about ${tbsp} tablespoons (15 mL each)`
}

/** "about 1 in 36" for a percentage, for people who don't read percentages easily. */
export function shareInWords(pct: number): string {
  if (pct <= 0) return 'none'
  if (pct >= 50) return `about ${Math.round(pct / 10)} in 10`
  return `about 1 in ${Math.max(2, Math.round(100 / pct))}`
}

export const fmt = (x: number, digits = 1) => (x >= 100 ? x.toFixed(0) : x.toFixed(digits))
