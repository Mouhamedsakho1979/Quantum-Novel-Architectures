import os

# Contenu du fichier de configuration (Mode Sombre + Pro)
config_content = """[theme]
base="dark"
primaryColor="#00FF00"
backgroundColor="#0E1117"
secondaryBackgroundColor="#262730"
textColor="#FAFAFA"
font="sans serif"

[client]
toolbarMode="viewer"
"""

def create_config():
    # 1. Créer le dossier .streamlit s'il n'existe pas
    if not os.path.exists(".streamlit"):
        os.makedirs(".streamlit")
        print("📁 Dossier '.streamlit' créé.")
    
    # 2. Écrire le fichier config.toml
    with open(".streamlit/config.toml", "w") as f:
        f.write(config_content)
    
    print("✅ Fichier 'config.toml' généré avec succès !")
    print("🎨 Le mode sombre est maintenant forcé.")

if __name__ == "__main__":
    create_config()