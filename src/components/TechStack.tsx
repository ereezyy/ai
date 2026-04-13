const categories = [
  {
    label: 'Deep Learning',
    color: 'var(--primary)',
    items: ['TensorFlow 2.10+', 'PyTorch 1.12+', 'HuggingFace Transformers', 'scikit-learn'],
  },
  {
    label: 'Hyperparameter Tuning',
    color: 'var(--accent)',
    items: ['Ray Tune', 'Optuna', 'Hyperopt', 'MLflow'],
  },
  {
    label: 'Cloud & Deployment',
    color: 'var(--green)',
    items: ['AWS boto3', 'Azure Blob', 'GCP Storage', 'FastAPI + Uvicorn'],
  },
  {
    label: 'AI & Reasoning',
    color: 'var(--primary)',
    items: ['Groq API', 'ChromaDB', 'Gradio', 'Weights & Biases'],
  },
  {
    label: 'Visualization & UI',
    color: 'var(--accent)',
    items: ['Streamlit', 'Plotly', 'Matplotlib', 'Jupyter Lab'],
  },
  {
    label: 'Core Infrastructure',
    color: 'var(--green)',
    items: ['Python 3.9+', 'Click CLI', 'NumPy', 'pandas'],
  },
]

export default function TechStack() {
  return (
    <section style={styles.section}>
      <div style={styles.container}>
        <div style={styles.header}>
          <span style={styles.sectionTag}>TECH STACK</span>
          <h2 style={styles.heading}>Powered By The Best</h2>
          <p style={styles.subheading}>
            Industry-leading libraries across every layer of the stack.
          </p>
        </div>

        <div style={styles.grid}>
          {categories.map((cat, i) => (
            <div key={i} style={styles.card}>
              <div style={{ ...styles.colorBar, background: cat.color }} />
              <div style={styles.cardContent}>
                <span style={{ ...styles.catLabel, color: cat.color }}>{cat.label}</span>
                <ul style={styles.list}>
                  {cat.items.map((item, j) => (
                    <li key={j} style={styles.listItem}>
                      <span style={{ ...styles.bullet, background: cat.color }} />
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
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
    maxWidth: '1200px',
    margin: '0 auto',
  },
  header: {
    textAlign: 'center',
    marginBottom: '64px',
  },
  sectionTag: {
    fontFamily: 'var(--font-mono)',
    fontSize: '11px',
    letterSpacing: '0.2em',
    color: 'var(--green)',
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
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
    gap: '16px',
  },
  card: {
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
  },
  colorBar: {
    height: '3px',
    width: '100%',
    opacity: 0.8,
  },
  cardContent: {
    padding: '24px',
  },
  catLabel: {
    fontFamily: 'var(--font-mono)',
    fontSize: '11px',
    fontWeight: 700,
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
    display: 'block',
    marginBottom: '16px',
  },
  list: {
    listStyle: 'none',
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
  },
  listItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    fontFamily: 'var(--font-sans)',
    fontSize: '14px',
    color: 'var(--text-secondary)',
  },
  bullet: {
    width: '5px',
    height: '5px',
    borderRadius: '50%',
    flexShrink: 0,
    opacity: 0.7,
  },
}
