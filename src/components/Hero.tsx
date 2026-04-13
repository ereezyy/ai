import { useEffect, useState } from 'react'
import { Zap } from 'lucide-react'

const phrases = [
  'FORGE YOUR MODELS.',
  'SHATTER REALITY.',
  'ACHIEVE OMNIPOTENCE.',
  'DOMINATE THE STACK.',
]

export default function Hero() {
  const [phraseIdx, setPhraseIdx] = useState(0)
  const [displayed, setDisplayed] = useState('')
  const [typing, setTyping] = useState(true)

  useEffect(() => {
    const phrase = phrases[phraseIdx]
    if (typing) {
      if (displayed.length < phrase.length) {
        const t = setTimeout(() => setDisplayed(phrase.slice(0, displayed.length + 1)), 45)
        return () => clearTimeout(t)
      } else {
        const t = setTimeout(() => setTyping(false), 1800)
        return () => clearTimeout(t)
      }
    } else {
      if (displayed.length > 0) {
        const t = setTimeout(() => setDisplayed(displayed.slice(0, -1)), 25)
        return () => clearTimeout(t)
      } else {
        setPhraseIdx((i) => (i + 1) % phrases.length)
        setTyping(true)
      }
    }
  }, [displayed, typing, phraseIdx])

  return (
    <section style={styles.section}>
      <div style={styles.gridOverlay} aria-hidden />
      <div style={styles.glowOrb} aria-hidden />

      <div style={styles.badge}>
        <Zap size={12} color="var(--accent)" />
        <span style={styles.badgeText}>v1.0.0 — LIVE</span>
        <span style={styles.badgeDot} />
      </div>

      <h1 style={styles.title}>
        AI<span style={styles.titleAccent}>TOOLKIT</span>
      </h1>

      <p style={styles.subtitle}>THE OMNIPOTENT FORGE</p>

      <div style={styles.typewriterWrap}>
        <span style={styles.typewriter}>{displayed}</span>
        <span style={styles.cursor}>|</span>
      </div>

      <p style={styles.desc}>
        A god-tier machine learning development suite. Build, train, evaluate,
        and deploy AI models across every platform — all from a single unified toolkit.
      </p>

      <div style={styles.actions}>
        <a href="#modules" style={styles.btnPrimary}>Explore Modules</a>
        <a href="#cli" style={styles.btnSecondary}>CLI Reference</a>
      </div>

      <div style={styles.authorRow}>
        <span style={styles.authorLabel}>by</span>
        <span style={styles.author}>ereezyy</span>
      </div>
    </section>
  )
}

const styles: Record<string, React.CSSProperties> = {
  section: {
    position: 'relative',
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    textAlign: 'center',
    padding: '80px 24px',
    overflow: 'hidden',
  },
  gridOverlay: {
    position: 'absolute',
    inset: 0,
    backgroundImage: `
      linear-gradient(rgba(0, 212, 255, 0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0, 212, 255, 0.03) 1px, transparent 1px)
    `,
    backgroundSize: '60px 60px',
    pointerEvents: 'none',
  },
  glowOrb: {
    position: 'absolute',
    top: '20%',
    left: '50%',
    transform: 'translateX(-50%)',
    width: '600px',
    height: '600px',
    borderRadius: '50%',
    background: 'radial-gradient(circle, rgba(0,212,255,0.06) 0%, transparent 70%)',
    pointerEvents: 'none',
  },
  badge: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    background: 'var(--bg-elevated)',
    border: '1px solid var(--accent-border)',
    borderRadius: '99px',
    padding: '6px 14px',
    marginBottom: '32px',
    position: 'relative',
    zIndex: 1,
  },
  badgeText: {
    fontFamily: 'var(--font-mono)',
    fontSize: '11px',
    color: 'var(--accent)',
    letterSpacing: '0.08em',
    fontWeight: 700,
  },
  badgeDot: {
    width: '6px',
    height: '6px',
    borderRadius: '50%',
    background: 'var(--green)',
    boxShadow: '0 0 6px var(--green)',
    animation: 'pulse-glow 2s ease-in-out infinite',
  },
  title: {
    fontFamily: 'var(--font-sans)',
    fontSize: 'clamp(56px, 10vw, 120px)',
    fontWeight: 700,
    letterSpacing: '-0.03em',
    lineHeight: 1,
    color: 'var(--text-primary)',
    position: 'relative',
    zIndex: 1,
  },
  titleAccent: {
    color: 'var(--primary)',
    textShadow: '0 0 40px rgba(0,212,255,0.4)',
    marginLeft: '8px',
  },
  subtitle: {
    fontFamily: 'var(--font-mono)',
    fontSize: '13px',
    letterSpacing: '0.25em',
    color: 'var(--text-muted)',
    marginTop: '8px',
    marginBottom: '32px',
    position: 'relative',
    zIndex: 1,
  },
  typewriterWrap: {
    minHeight: '40px',
    display: 'flex',
    alignItems: 'center',
    gap: '2px',
    marginBottom: '24px',
    position: 'relative',
    zIndex: 1,
  },
  typewriter: {
    fontFamily: 'var(--font-mono)',
    fontSize: 'clamp(18px, 2.5vw, 26px)',
    fontWeight: 700,
    color: 'var(--primary)',
    letterSpacing: '0.05em',
  },
  cursor: {
    fontFamily: 'var(--font-mono)',
    fontSize: 'clamp(18px, 2.5vw, 26px)',
    fontWeight: 700,
    color: 'var(--primary)',
    animation: 'blink 1s step-end infinite',
  },
  desc: {
    maxWidth: '560px',
    fontSize: '17px',
    lineHeight: 1.7,
    color: 'var(--text-secondary)',
    marginBottom: '40px',
    position: 'relative',
    zIndex: 1,
  },
  actions: {
    display: 'flex',
    gap: '16px',
    flexWrap: 'wrap',
    justifyContent: 'center',
    position: 'relative',
    zIndex: 1,
  },
  btnPrimary: {
    fontFamily: 'var(--font-sans)',
    fontWeight: 700,
    fontSize: '14px',
    letterSpacing: '0.04em',
    color: '#000',
    background: 'var(--primary)',
    border: 'none',
    borderRadius: 'var(--radius-sm)',
    padding: '14px 32px',
    textDecoration: 'none',
    cursor: 'pointer',
    transition: 'all 0.2s',
    boxShadow: '0 0 20px rgba(0,212,255,0.3)',
  },
  btnSecondary: {
    fontFamily: 'var(--font-sans)',
    fontWeight: 700,
    fontSize: '14px',
    letterSpacing: '0.04em',
    color: 'var(--text-primary)',
    background: 'transparent',
    border: '1px solid var(--border-bright)',
    borderRadius: 'var(--radius-sm)',
    padding: '14px 32px',
    textDecoration: 'none',
    cursor: 'pointer',
    transition: 'all 0.2s',
  },
  authorRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginTop: '48px',
    position: 'relative',
    zIndex: 1,
  },
  authorLabel: {
    fontSize: '13px',
    color: 'var(--text-muted)',
    fontFamily: 'var(--font-mono)',
  },
  author: {
    fontSize: '13px',
    fontFamily: 'var(--font-mono)',
    fontWeight: 700,
    color: 'var(--accent)',
    letterSpacing: '0.05em',
  },
}
