import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from config import Config
from utils.database import db
from controllers.main_controller import main

def create_app(config_class=Config):
    """Application factory function"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions
    db.init_app(app)
    
    # Register blueprints
    app.register_blueprint(main)
    
    # Add template functions
    @app.context_processor
    def utility_processor():
        def now():
            return datetime.now()
        
        return dict(now=now, config=app.config)
    
    # Create data directories if they don't exist
    os.makedirs(app.config['RAW_DATA_DIR'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_DATA_DIR'], exist_ok=True)
    os.makedirs(app.config['EXTERNAL_DATA_DIR'], exist_ok=True)
    os.makedirs(app.config['MODELS_DIR'], exist_ok=True)
    
    return app

# Create the application instance
app = create_app()

# Create error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_code=404, error_message="Page Not Found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', error_code=500, error_message="Internal Server Error"), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))