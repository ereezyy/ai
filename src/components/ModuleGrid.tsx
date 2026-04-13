import { useState } from 'react'
import {
  Database,
  Layers,
  Dumbbell,
  BarChart2,
  Rocket,
  Wand2,
  MessageSquare,
  Bot,
  Brain,
  FolderOpen,
} from 'lucide-react'

const modules = [
  {
    icon: Database,
    name: 'data.py',
    title: 'Data Processor',
    tag: 'INGESTION',
    color: 'var(--primary)',
    desc: 'Load and preprocess datasets from CSV, JSON, Parquet, and more. Handles normalization, augmentation, and pipeline construction.',
    api: ['DataProcessor', 'load_dataset()'],
    size: 'normal',
  },
  {
    icon: Layers,
    name: 'models.py',
    title: 'Model Builder',
    tag: 'ARCHITECTURE',
    color: 'var(--accent)',
    desc: 'Construct image classifiers, text classifiers, and time series models. Wraps TensorFlow and PyTorch under a unified interface.',
    api: ['ModelBuilder', 'PretrainedModels', 'create_image_classifier()', 'create_text_classifier()'],
    size: 'normal',
  },
  {
    icon: Dumbbell,
    name: 'training.py',
    title: 'Trainer',
    tag: 'OPTIMIZATION',
    color: 'var(--green)',
    desc: 'Full training loop management with configurable epochs, batch sizes, learning rates, and optimizer strategies.',
    api: ['Trainer', 'trainer.fit()'],
    size: 'normal',
  },
  {
    icon: BarChart2,
    name: 'evaluation.py',
    title: 'Evaluator',
    tag: 'METRICS',
    color: 'var(--primary)',
    desc: 'Compute accuracy, precision, recall, F1-score, and custom metrics. Generate full evaluation reports for model inspection.',
    api: ['Evaluator', 'evaluator.evaluate()'],
    size: 'normal',
  },
  {
    icon: Rocket,
    name: 'deployment.py',
    title: 'Model Deployer',
    tag: 'DEPLOYMENT',
    color: 'var(--accent)',
    desc: 'Ship models to AWS, Azure, GCP, or local endpoints. Automated containerization and serving infrastructure.',
    api: ['ModelDeployer', 'deployer.deploy()'],
    size: 'wide',
  },
  {
    icon: Wand2,
    name: 'automl.py',
    title: 'AutoML Pipeline',
    tag: 'AUTOMATION',
    color: 'var(--green)',
    desc: 'Automated hyperparameter tuning powered by Ray Tune, Optuna, and Hyperopt. Let the machine find the optimal configuration.',
    api: ['AutoMLPipeline'],
    size: 'normal',
  },
  {
    icon: MessageSquare,
    name: 'nlp.py',
    title: 'NLP Engine',
    tag: 'LANGUAGE',
    color: 'var(--primary)',
    desc: 'Natural language processing utilities with Groq API integration for fast intent parsing and command interpretation.',
    api: ['Groq API', 'Intent Parsing', 'HuggingFace Transformers'],
    size: 'normal',
  },
  {
    icon: Bot,
    name: 'autonomy.py',
    title: 'Autonomy Engine',
    tag: 'AUTONOMOUS',
    color: 'var(--accent)',
    desc: 'Autonomous AI execution engine. Full system override capabilities with self-directed task planning and execution.',
    api: ['AutonomyEngine', 'god-mode CLI'],
    size: 'normal',
  },
  {
    icon: Brain,
    name: 'skills.py',
    title: 'Skill Acquisition',
    tag: 'EVOLUTION',
    color: 'var(--green)',
    desc: 'Learn new skills dynamically from GitHub repositories and web search. Evolutionary personality tracking over time.',
    api: ['learn-skill', 'evolve', 'ChromaDB vector store'],
    size: 'normal',
  },
  {
    icon: FolderOpen,
    name: 'utils/project.py',
    title: 'Project Manager',
    tag: 'SCAFFOLDING',
    color: 'var(--primary)',
    desc: 'Scaffold new AI projects with opinionated directory structure. Templates for classification, generation, and time series tasks.',
    api: ['ProjectManager', 'create_project()'],
    size: 'normal',
  },
]

export default function ModuleGrid() {
  const [hovered, setHovered] = useState<number | null>(null)

  return (
    <section id="modules" style={styles.section}>
      <div style={styles.container}>
        <div style={styles.header}>
          <span style={styles.sectionTag}>MODULES</span>
          <h2 style={styles.heading}>The Full Arsenal</h2>
          <p style={styles.subheading}>
            10 specialized modules covering every stage of the ML lifecycle.
          </p>
        </div>

        <div style={styles.grid}>
          {modules.map((mod, i) => (
            <div
              key={i}
              style={{
                ...styles.card,
                ...(mod.size === 'wide' ? styles.cardWide : {}),
                ...(hovered === i ? styles.cardHovered : {}),
                borderColor: hovered === i ? mod.color + '44' : 'var(--border)',
              }}
              onMouseEnter={() => setHovered(i)}
              onMouseLeave={() => setHovered(null)}
            >
              <div style={styles.cardTop}>
                <div style={{ ...styles.iconWrap, background: mod.color + '18', boxShadow: hovered === i ? `0 0 16px ${mod.color}22` : 'none' }}>
                  <mod.icon size={20} color={mod.color} />
                </div>
                <span style={{ ...styles.tag, color: mod.color, background: mod.color + '12', borderColor: mod.color + '28' }}>{mod.tag}</span>
              </div>

              <code style={styles.fileName}>{mod.name}</code>
              <h3 style={styles.cardTitle}>{mod.title}</h3>
              <p style={styles.cardDesc}>{mod.desc}</p>

              <div style={styles.apiList}>
                {mod.api.map((a, j) => (
                  <span key={j} style={styles.apiChip}>{a}</span>
                ))}
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
    padding: '96px 24px',
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
    color: 'var(--primary)',
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
    maxWidth: '480px',
    margin: '0 auto',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
    gap: '16px',
  },
  card: {
    background: 'var(--bg-card)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    padding: '28px',
    cursor: 'default',
    transition: 'all 0.25s ease',
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  cardWide: {
    gridColumn: 'span 2',
  },
  cardHovered: {
    background: 'var(--bg-card-hover)',
    transform: 'translateY(-2px)',
    boxShadow: '0 12px 40px rgba(0,0,0,0.4)',
  },
  cardTop: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: '4px',
  },
  iconWrap: {
    width: '40px',
    height: '40px',
    borderRadius: 'var(--radius-sm)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'box-shadow 0.25s',
  },
  tag: {
    fontFamily: 'var(--font-mono)',
    fontSize: '10px',
    fontWeight: 700,
    letterSpacing: '0.12em',
    padding: '4px 10px',
    borderRadius: '4px',
    border: '1px solid',
  },
  fileName: {
    fontFamily: 'var(--font-mono)',
    fontSize: '12px',
    color: 'var(--text-muted)',
    letterSpacing: '0.02em',
  },
  cardTitle: {
    fontFamily: 'var(--font-sans)',
    fontSize: '20px',
    fontWeight: 700,
    color: 'var(--text-primary)',
    letterSpacing: '-0.01em',
  },
  cardDesc: {
    fontSize: '14px',
    lineHeight: 1.65,
    color: 'var(--text-secondary)',
    flexGrow: 1,
  },
  apiList: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '6px',
    marginTop: '4px',
  },
  apiChip: {
    fontFamily: 'var(--font-mono)',
    fontSize: '11px',
    color: 'var(--text-muted)',
    background: 'var(--bg-elevated)',
    border: '1px solid var(--border)',
    borderRadius: '4px',
    padding: '3px 8px',
  },
}
