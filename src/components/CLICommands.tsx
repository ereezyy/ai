import { useState } from 'react'
import { Terminal, Copy, Check } from 'lucide-react'

const commands = [
  { cmd: 'ai-toolkit create-project', args: '<name>', desc: 'Scaffold a new AI project with organized directory structure' },
  { cmd: 'ai-toolkit preprocess', args: '--input data.csv --output processed/', desc: 'Load and preprocess raw datasets for training' },
  { cmd: 'ai-toolkit train', args: '--model resnet50 --epochs 50', desc: 'Launch a training run with full progress tracking' },
  { cmd: 'ai-toolkit evaluate', args: '--model ./models/best.pt', desc: 'Evaluate model performance and generate metric reports' },
  { cmd: 'ai-toolkit deploy', args: '--platform aws --model ./models/prod.pt', desc: 'Deploy model to AWS, Azure, GCP, or local endpoint' },
  { cmd: 'ai-toolkit predict', args: '--input image.jpg', desc: 'Run inference on new data using a trained model' },
  { cmd: 'ai-toolkit learn-skill', args: '"computer vision segmentation"', desc: 'Acquire new skills from GitHub repos and web search' },
  { cmd: 'ai-toolkit evolve', args: '', desc: 'Trigger personality and capability evolution cycle' },
  { cmd: 'ai-toolkit awaken', args: '--goal "build classification pipeline"', desc: 'Activate autonomous mode with Groq-powered reasoning' },
  { cmd: 'ai-toolkit god-mode', args: '', desc: 'Enter interactive autonomous CLI with full system access' },
  { cmd: 'ai-toolkit jupyter', args: '', desc: 'Launch JupyterLab in the current project directory' },
  { cmd: 'ai-toolkit dashboard', args: '', desc: 'Open the Streamlit monitoring dashboard' },
]

export default function CLICommands() {
  const [copied, setCopied] = useState<number | null>(null)

  const copy = (i: number, text: string) => {
    navigator.clipboard.writeText(text)
    setCopied(i)
    setTimeout(() => setCopied(null), 1800)
  }

  return (
    <section id="cli" style={styles.section}>
      <div style={styles.container}>
        <div style={styles.header}>
          <span style={styles.sectionTag}>CLI REFERENCE</span>
          <h2 style={styles.heading}>Command the Machine</h2>
          <p style={styles.subheading}>
            12 commands. Full control over the entire ML pipeline from a single terminal.
          </p>
        </div>

        <div style={styles.terminal}>
          <div style={styles.terminalHeader}>
            <div style={styles.terminalDots}>
              <span style={{ ...styles.dot, background: '#ff5f56' }} />
              <span style={{ ...styles.dot, background: '#ffbd2e' }} />
              <span style={{ ...styles.dot, background: '#27c93f' }} />
            </div>
            <span style={styles.terminalTitle}>
              <Terminal size={12} />
              ai-toolkit — bash
            </span>
          </div>

          <div style={styles.commandList}>
            {commands.map((c, i) => (
              <div key={i} style={styles.commandRow}>
                <div style={styles.commandLine}>
                  <span style={styles.prompt}>$</span>
                  <span style={styles.cmd}>{c.cmd}</span>
                  {c.args && <span style={styles.args}>{c.args}</span>}
                  <button
                    style={styles.copyBtn}
                    onClick={() => copy(i, `${c.cmd}${c.args ? ' ' + c.args : ''}`)}
                    title="Copy command"
                  >
                    {copied === i ? <Check size={12} color="var(--green)" /> : <Copy size={12} />}
                  </button>
                </div>
                <p style={styles.cmdDesc}>{c.desc}</p>
              </div>
            ))}
          </div>
        </div>

        <div style={styles.installBox}>
          <span style={styles.installLabel}>INSTALL</span>
          <code style={styles.installCmd}>pip install ai-toolkit</code>
          <button
            style={styles.copyBtn}
            onClick={() => copy(999, 'pip install ai-toolkit')}
            title="Copy"
          >
            {copied === 999 ? <Check size={14} color="var(--green)" /> : <Copy size={14} />}
          </button>
        </div>
      </div>
    </section>
  )
}

const styles: Record<string, React.CSSProperties> = {
  section: {
    padding: '0 24px 96px',
  },
  container: {
    maxWidth: '900px',
    margin: '0 auto',
  },
  header: {
    textAlign: 'center',
    marginBottom: '56px',
  },
  sectionTag: {
    fontFamily: 'var(--font-mono)',
    fontSize: '11px',
    letterSpacing: '0.2em',
    color: 'var(--accent)',
    display: 'block',
    marginBottom: '12px',
  },
  heading: {
    fontSize: 'clamp(32px, 5vw, 48px)',
    fontWeight: 700,
    letterSpacing: '-0.02em',
    color: 'var(--text-primary)',
    marginBottom: '16px',
  },
  subheading: {
    fontSize: '17px',
    color: 'var(--text-secondary)',
  },
  terminal: {
    background: '#0c0e15',
    border: '1px solid var(--border-bright)',
    borderRadius: 'var(--radius-lg)',
    overflow: 'hidden',
    marginBottom: '24px',
  },
  terminalHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    padding: '14px 20px',
    borderBottom: '1px solid var(--border)',
    background: '#0f1118',
  },
  terminalDots: {
    display: 'flex',
    gap: '6px',
  },
  dot: {
    width: '10px',
    height: '10px',
    borderRadius: '50%',
    display: 'block',
  },
  terminalTitle: {
    fontFamily: 'var(--font-mono)',
    fontSize: '12px',
    color: 'var(--text-muted)',
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    flex: 1,
    justifyContent: 'center',
  },
  commandList: {
    padding: '8px 0',
  },
  commandRow: {
    padding: '10px 20px',
    borderBottom: '1px solid rgba(255,255,255,0.03)',
  },
  commandLine: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginBottom: '4px',
  },
  prompt: {
    fontFamily: 'var(--font-mono)',
    fontSize: '13px',
    color: 'var(--green)',
    userSelect: 'none',
  },
  cmd: {
    fontFamily: 'var(--font-mono)',
    fontSize: '13px',
    color: 'var(--primary)',
    fontWeight: 700,
  },
  args: {
    fontFamily: 'var(--font-mono)',
    fontSize: '13px',
    color: 'var(--accent)',
  },
  copyBtn: {
    background: 'transparent',
    border: 'none',
    cursor: 'pointer',
    color: 'var(--text-muted)',
    padding: '2px 4px',
    display: 'flex',
    alignItems: 'center',
    marginLeft: 'auto',
    transition: 'color 0.15s',
  },
  cmdDesc: {
    fontFamily: 'var(--font-sans)',
    fontSize: '12px',
    color: 'var(--text-muted)',
    marginLeft: '20px',
  },
  installBox: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
    background: 'var(--bg-elevated)',
    border: '1px solid var(--primary-border)',
    borderRadius: 'var(--radius)',
    padding: '16px 24px',
  },
  installLabel: {
    fontFamily: 'var(--font-mono)',
    fontSize: '10px',
    fontWeight: 700,
    letterSpacing: '0.15em',
    color: 'var(--primary)',
    background: 'var(--primary-dim)',
    border: '1px solid var(--primary-border)',
    borderRadius: '4px',
    padding: '3px 8px',
    whiteSpace: 'nowrap',
  },
  installCmd: {
    fontFamily: 'var(--font-mono)',
    fontSize: '14px',
    color: 'var(--text-primary)',
    flex: 1,
  },
}
