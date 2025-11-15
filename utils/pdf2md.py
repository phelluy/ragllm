"""
Utilitaire de conversion PDF vers Markdown utilisant l'API OCR de Mistral.

Ce script convertit des fichiers PDF en Markdown en utilisant l'API OCR de Mistral.
Il gère également le téléchargement automatique des images détectées dans les PDF.

Fonctionnalités:
- Conversion de PDF en Markdown via l'API Mistral OCR
- Téléchargement automatique des images depuis les serveurs Mistral
- Organisation des images dans des sous-dossiers dédiés (format: {nom_pdf}_images/)
- Mise à jour des liens markdown pour référencer les images locales

Usage:
    python pdf2md.py <dossier>

Exemple:
    python pdf2md.py ./mon_dossier_pdf

Prérequis:
- Variable d'environnement MISTRAL_API_KEY définie
- Fichiers PDF à convertir

Structure de sortie:
    mon_dossier/
    ├── document.pdf
    ├── document.md              # Markdown généré
    └── document_images/         # Images téléchargées
        ├── image_001.png
        ├── image_002.png
        └── ...
"""
import os
import sys
import json
import requests
import re
from urllib.parse import urlparse

API_KEY = os.environ.get("MISTRAL_API_KEY")  # Stocke ta clé API dans une variable d'environnement

def upload_file(pdf_path):
    with open(pdf_path, 'rb') as f:
        files = {
            'purpose': (None, 'ocr'),
            'file': (os.path.basename(pdf_path), f, 'application/pdf')
        }
        headers = {"Authorization": f"Bearer {API_KEY}"}
        response = requests.post("https://api.mistral.ai/v1/files", headers=headers, files=files)
    if response.status_code != 200:
        print(f"Upload failed for {pdf_path}: {response.text}")
        return None
    return response.json().get("id")

def get_signed_url(file_id):
    url = f"https://api.mistral.ai/v1/files/{file_id}/url?expiry=24"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to get signed URL: {response.text}")
        return None
    return response.json().get("url")

def run_ocr(document_url):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-ocr-latest",
        "include_image_base64": False,
        "document": {
            "type": "document_url",
            "document_url": document_url
        }
    }
    response = requests.post("https://api.mistral.ai/v1/ocr", headers=headers, data=json.dumps(payload))
    if response.status_code != 200:
        print(f"OCR failed: {response.text}")
        return None
    return response.json()

def download_image(image_url, output_path):
    """Télécharge une image depuis l'URL Mistral et la sauvegarde localement."""
    try:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        response = requests.get(image_url, headers=headers, stream=True)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        else:
            print(f"Failed to download image from {image_url}: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading image from {image_url}: {e}")
        return False

def extract_image_urls(markdown_text):
    """Extrait toutes les URLs d'images du markdown."""
    # Recherche les images au format ![alt](url)
    pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'
    matches = re.findall(pattern, markdown_text)
    return [(alt, url) for alt, url in matches]

def process_images(markdown, images_dir, pdf_name):
    """
    Télécharge les images référencées dans le markdown et met à jour les liens.
    
    Args:
        markdown: Le texte markdown contenant des références d'images
        images_dir: Le répertoire où sauvegarder les images
        pdf_name: Le nom du PDF (sans extension) pour nommer le sous-dossier
    
    Returns:
        Le markdown avec les liens d'images mis à jour
    """
    # Créer le sous-dossier pour les images de ce PDF
    pdf_images_dir = os.path.join(images_dir, f"{pdf_name}_images")
    os.makedirs(pdf_images_dir, exist_ok=True)
    
    # Extraire les URLs d'images
    image_refs = extract_image_urls(markdown)
    
    if not image_refs:
        return markdown
    
    updated_markdown = markdown
    image_counter = 1
    
    for alt_text, image_url in image_refs:
        # Déterminer l'extension de l'image depuis l'URL ou utiliser .png par défaut
        parsed_url = urlparse(image_url)
        path_parts = parsed_url.path.split('.')
        extension = f".{path_parts[-1]}" if len(path_parts) > 1 and len(path_parts[-1]) <= 4 else ".png"
        
        # Nom de fichier local pour l'image
        local_image_name = f"image_{image_counter:03d}{extension}"
        local_image_path = os.path.join(pdf_images_dir, local_image_name)
        
        # Télécharger l'image
        print(f"  Downloading image {image_counter}: {image_url}")
        if download_image(image_url, local_image_path):
            # Mettre à jour le lien dans le markdown (chemin relatif)
            relative_path = os.path.join(f"{pdf_name}_images", local_image_name)
            updated_markdown = updated_markdown.replace(
                f"![{alt_text}]({image_url})",
                f"![{alt_text}]({relative_path})"
            )
            print(f"  Saved as {relative_path}")
            image_counter += 1
        else:
            print(f"  Failed to download image from {image_url}")
    
    return updated_markdown

def extract_markdown(ocr_json):
    if "pages" not in ocr_json or not ocr_json["pages"]:
        return ""
    return "\n".join(page.get("markdown", "") for page in ocr_json["pages"])

def process_pdf(pdf_path, md_path):
    print(f"Processing {pdf_path}")
    file_id = upload_file(pdf_path)
    if not file_id:
        return

    doc_url = get_signed_url(file_id)
    if not doc_url:
        return

    ocr_result = run_ocr(doc_url)
    if not ocr_result:
        return

    markdown = extract_markdown(ocr_result)
    
    # Traiter les images si présentes dans le markdown
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    images_dir = os.path.dirname(md_path)
    markdown = process_images(markdown, images_dir, pdf_name)
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"Wrote {md_path}")

def main(root_dir):
    if not API_KEY:
        print("❌ API key manquante. Définis MISTRAL_API_KEY dans les variables d’environnement.")
        sys.exit(1)

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(dirpath, filename)
                md_path = os.path.splitext(pdf_path)[0] + ".md"
                if os.path.exists(md_path):
                    print(f"Skip (exists): {md_path}")
                    continue
                process_pdf(pdf_path, md_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pdf2md.py <dossier>")
        sys.exit(1)
    main(sys.argv[1])
