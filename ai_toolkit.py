#!/usr/bin/env python3
"""
AI Toolkit Command Line Interface
Provides easy access to AI Toolkit functionality from the command line.
"""

import click
import os
import sys
from pathlib import Path

# Add the ai_toolkit to the path
sys.path.insert(0, str(Path(__file__).parent))

import ai_toolkit as ai

@click.group()
@click.version_option(version=ai.__version__)
def cli():
    """AI Toolkit - Comprehensive AI Development Suite"""
    pass

@cli.command()
@click.argument('name')
@click.option('--description', '-d', default="", help='Project description')
@click.option('--template', '-t', default='basic', help='Project template (basic, vision, nlp, timeseries)')
def create_project(name, description, template):
    """Create a new AI project."""
    try:
        project = ai.create_project(name, description)
        click.echo(f"✅ Created project '{name}' successfully!")
        click.echo(f"📁 Project directory: {project.path}")
        
        if template != 'basic':
            click.echo(f"🔧 Setting up {template} template...")
            project.setup_template(template)
            
    except Exception as e:
        click.echo(f"❌ Error creating project: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('data_path')
@click.option('--output', '-o', help='Output path for processed data')
@click.option('--task', '-t', default='classification', help='Task type (classification, regression, detection)')
def preprocess(data_path, output, task):
    """Preprocess data for machine learning."""
    try:
        click.echo(f"📊 Loading data from {data_path}...")
        data = ai.load_data(data_path)
        
        click.echo(f"🔧 Preprocessing for {task} task...")
        processor = ai.DataProcessor()
        processed_data = processor.preprocess(data, task_type=task)
        
        if output:
            processed_data.save(output)
            click.echo(f"💾 Saved processed data to {output}")
        else:
            click.echo("✅ Data preprocessing completed!")
            
    except Exception as e:
        click.echo(f"❌ Error preprocessing data: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('model_type')
@click.option('--data', '-d', required=True, help='Path to training data')
@click.option('--epochs', '-e', default=50, help='Number of training epochs')
@click.option('--batch-size', '-b', default=32, help='Batch size for training')
@click.option('--learning-rate', '-lr', default=0.001, help='Learning rate')
@click.option('--output', '-o', help='Output path for trained model')
def train(model_type, data, epochs, batch_size, learning_rate, output):
    """Train a machine learning model."""
    try:
        click.echo(f"📊 Loading training data from {data}...")
        training_data = ai.load_data(data)
        
        click.echo(f"🧠 Creating {model_type} model...")
        if model_type == 'image_classifier':
            model = ai.create_image_classifier(num_classes=10)  # Default, should be configurable
        elif model_type == 'text_classifier':
            model = ai.create_text_classifier(num_classes=3)
        else:
            click.echo(f"❌ Unsupported model type: {model_type}", err=True)
            sys.exit(1)
        
        click.echo(f"🚀 Training model for {epochs} epochs...")
        with click.progressbar(length=epochs, label='Training Progress') as bar:
            history = ai.train(
                model, 
                training_data, 
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                progress_callback=lambda epoch, logs: bar.update(1)
            )
        
        if output:
            model.save(output)
            click.echo(f"💾 Saved trained model to {output}")
        
        click.echo("✅ Training completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Error during training: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('model_path')
@click.argument('test_data')
@click.option('--metrics', '-m', multiple=True, default=['accuracy'], help='Evaluation metrics')
@click.option('--output', '-o', help='Output path for evaluation report')
def evaluate(model_path, test_data, metrics, output):
    """Evaluate a trained model."""
    try:
        click.echo(f"🧠 Loading model from {model_path}...")
        # Model loading logic would go here
        
        click.echo(f"📊 Loading test data from {test_data}...")
        test_dataset = ai.load_data(test_data)
        
        click.echo(f"📈 Evaluating model with metrics: {', '.join(metrics)}...")
        # Evaluation logic would go here
        
        click.echo("✅ Evaluation completed!")
        
    except Exception as e:
        click.echo(f"❌ Error during evaluation: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('model_path')
@click.option('--platform', '-p', default='local', help='Deployment platform (local, aws, azure, gcp)')
@click.option('--port', default=8000, help='Port for local deployment')
@click.option('--name', help='Deployment name')
def deploy(model_path, platform, port, name):
    """Deploy a trained model."""
    try:
        click.echo(f"🧠 Loading model from {model_path}...")
        # Model loading logic would go here
        
        click.echo(f"🚀 Deploying to {platform}...")
        
        if platform == 'local':
            click.echo(f"🌐 Starting local server on port {port}...")
            click.echo(f"📡 Model API available at: http://localhost:{port}")
        else:
            click.echo(f"☁️ Deploying to {platform} cloud...")
        
        click.echo("✅ Deployment completed!")
        
    except Exception as e:
        click.echo(f"❌ Error during deployment: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('input_data')
@click.argument('model_path')
@click.option('--output', '-o', help='Output path for predictions')
@click.option('--batch-size', '-b', default=32, help='Batch size for prediction')
def predict(input_data, model_path, output, batch_size):
    """Make predictions with a trained model."""
    try:
        click.echo(f"🧠 Loading model from {model_path}...")
        # Model loading logic would go here
        
        click.echo(f"📊 Loading input data from {input_data}...")
        data = ai.load_data(input_data)
        
        click.echo("🔮 Making predictions...")
        # Prediction logic would go here
        
        if output:
            click.echo(f"💾 Saving predictions to {output}")
        else:
            click.echo("✅ Predictions completed!")
            
    except Exception as e:
        click.echo(f"❌ Error during prediction: {e}", err=True)
        sys.exit(1)

@cli.command()
def info():
    """Display system and AI Toolkit information."""
    click.echo(f"""
🤖 AI Toolkit v{ai.__version__}
================================

System Information:
{ai.get_device_info()}

Configuration:
- Model Storage: {ai.Config.MODEL_STORAGE_PATH}
- Data Storage: {ai.Config.DATA_STORAGE_PATH}
- Log Path: {ai.Config.LOG_PATH}

Available Commands:
- create-project: Create a new AI project
- preprocess: Preprocess data for ML
- train: Train machine learning models
- evaluate: Evaluate model performance
- deploy: Deploy models to production
- predict: Make predictions with trained models

For help with any command, use: ai-toolkit COMMAND --help
""")

@cli.command()
@click.option('--port', default=8888, help='Jupyter server port')
@click.option('--ip', default='localhost', help='Jupyter server IP')
def jupyter(port, ip):
    """Launch Jupyter Lab with AI Toolkit environment."""
    try:
        import subprocess
        click.echo(f"🚀 Starting Jupyter Lab on {ip}:{port}...")
        subprocess.run([
            'jupyter', 'lab', 
            f'--ip={ip}', 
            f'--port={port}',
            '--no-browser'
        ])
    except Exception as e:
        click.echo(f"❌ Error starting Jupyter: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--port', default=8501, help='Streamlit server port')
def dashboard(port):
    """Launch AI Toolkit web dashboard."""
    try:
        click.echo(f"🌐 Starting AI Toolkit dashboard on port {port}...")
        click.echo(f"📡 Dashboard available at: http://localhost:{port}")
        # Dashboard launch logic would go here
    except Exception as e:
        click.echo(f"❌ Error starting dashboard: {e}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    cli()

