import { Github, Zap } from 'lucide-react'

export default function Footer() {
  return (
    <footer style={styles.footer}>
      <div style={styles.line} />
      <div style={styles.inner}>
        <div style={styles.left}>
          <div style={styles.logo}>
            <Zap size={16} color="var(--primary)" />
            <span style={styles.logoText}>AI TOOLKIT</span>
          </div>
          <p style={styles.tagline}>The Omnipotent Forge. v1.0.0</p>
        </div>

        <div style={styles.right}>
          <a
            href="https://github.com/ereezyy/ai"
            target="_blank"
            rel="noopener noreferrer"
            style={styles.link}
          >
            <Github size={16} />
            ereezyy/ai
          </a>
          <span style={styles.sep}>·</span>
          <span style={styles.license}>MIT License</span>
          <span style={styles.sep}>·</span>
          <span style={styles.author}>by ereezyy</span>
        </div>
      </div>
    </footer>
  )
}

const styles: Record<string, React.CSSProperties> = {
  footer: {
    padding: '0 24px 48px',
  },
  line: {
    height: '1px',
    background: 'var(--border)',
    maxWidth: '1200px',
    margin: '0 auto 32px',
  },
  inner: {
    maxWidth: '1200px',
    margin: '0 auto',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
    gap: '16px',
  },
  left: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  logoText: {
    fontFamily: 'var(--font-mono)',
    fontSize: '14px',
    fontWeight: 700,
    color: 'var(--text-primary)',
    letterSpacing: '0.08em',
  },
  tagline: {
    fontFamily: 'var(--font-mono)',
    fontSize: '12px',
    color: 'var(--text-muted)',
  },
  right: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    flexWrap: 'wrap',
  },
  link: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    fontFamily: 'var(--font-mono)',
    fontSize: '13px',
    color: 'var(--text-secondary)',
    textDecoration: 'none',
    transition: 'color 0.15s',
  },
  sep: {
    color: 'var(--text-muted)',
    fontSize: '13px',
  },
  license: {
    fontFamily: 'var(--font-mono)',
    fontSize: '13px',
    color: 'var(--text-muted)',
  },
  author: {
    fontFamily: 'var(--font-mono)',
    fontSize: '13px',
    color: 'var(--accent)',
    fontWeight: 700,
  },
}
