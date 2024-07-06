# Librerias ----------------------------------------

import os, sys
import argparse
sys.path.append(os.getcwd()) # Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código
import platform 

sistema_operativo = platform.system()

    
# Definir extension de ejecutables ---------------------------------------- 

if sistema_operativo == 'Windows':
        extension_binarios = ".exe"
else:
        extension_binarios = ""

# Info ---------------------------------------- 

# Preproceso ---------------------------------------- 

os.system(f"python{extension_binarios} preprocessing/a01_preproceso.py")

os.system(f"python{extension_binarios} preprocessing/a02_preproceso_split.py")


# Modelo ---------------------------------------- 

os.system(f"python{extension_binarios} models/b01_creacion_modelos.py")