import sys
from pathlib import Path

# ensure the inner package folder is on sys.path
root = Path(__file__).resolve().parent / 'vision_desktop_app'
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

try:
    from vision_desktop_app.main import main
except Exception as e:
    print('Failed to import vision_desktop_app.main:', e)
    print('sys.path:', sys.path[0:5])
    raise

if __name__ == '__main__':
    main()
