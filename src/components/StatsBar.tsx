import { Cpu, Zap, Box, Cloud, GitBranch, Activity } from 'lucide-react'

const stats = [
  { icon: Cpu, label: 'Compute', value: 'GOD-TIER', color: 'var(--primary)' },
  { icon: Zap, label: 'Inference', value: 'INSTANT', color: 'var(--accent)' },
  { icon: Box, label: 'Modules', value: '10', color: 'var(--green)' },
  { icon: Cloud, label: 'Platforms', value: 'AWS · Azure · GCP', color: 'var(--primary)' },
  { icon: GitBranch, label: 'Version', value: '1.0.0', color: 'var(--accent)' },
  { icon: Activity, label: 'Status', value: 'ONLINE', color: 'var(--green)' },
]

export default function StatsBar() {
  return (
    <section style={styles.section}>
      <div style={styles.bar}>
        {stats.map((s, i) => (
          <div key={i} style={styles.stat}>
            <s.icon size={16} color={s.color} />
            <div style={styles.statContent}>
              <span style={{ ...styles.statValue, color: s.color }}>{s.value}</span>
              <span style={styles.statLabel}>{s.label}</span>
            </div>
            {i < stats.length - 1 && <div style={styles.divider} />}
          </div>
        ))}
      </div>
    </section>
  )
}

const styles: Record<string, React.CSSProperties> = {
  section: {
    padding: '0 24px',
    marginTop: '-1px',
  },
  bar: {
    maxWidth: '1200px',
    margin: '0 auto',
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '0',
    background: 'var(--bg-elevated)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    padding: '0 24px',
    overflow: 'hidden',
  },
  stat: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    padding: '20px 16px',
    flex: '1 1 0',
    minWidth: '140px',
    position: 'relative',
  },
  statContent: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
  },
  statValue: {
    fontFamily: 'var(--font-mono)',
    fontSize: '13px',
    fontWeight: 700,
    letterSpacing: '0.05em',
  },
  statLabel: {
    fontFamily: 'var(--font-sans)',
    fontSize: '11px',
    color: 'var(--text-muted)',
    letterSpacing: '0.05em',
    textTransform: 'uppercase',
  },
  divider: {
    position: 'absolute',
    right: 0,
    top: '20%',
    bottom: '20%',
    width: '1px',
    background: 'var(--border)',
  },
}
