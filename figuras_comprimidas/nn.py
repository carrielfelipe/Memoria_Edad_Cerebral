import os

# Directorio actual
directory = os.getcwd()

print(f"Directorio actual: {directory}")

# Recorrer todos los archivos en el directorio
for filename in os.listdir(directory):
    print(f"Encontrado archivo: {filename}")
    # Comprobar si el archivo contiene '-min'
    if '-min' in filename:
        # Crear el nuevo nombre del archivo eliminando '-min'
        new_filename = filename.replace('-min', '')
        # Obtener la ruta completa de los archivos
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        
        # Eliminar el archivo nuevo si ya existe
        if os.path.exists(new_file):
            os.remove(new_file)
            print(f"El archivo {new_filename} ya existía y fue eliminado.")
        
        # Renombrar el archivo
        os.rename(old_file, new_file)
        print(f'Renamed: {filename} -> {new_filename}')
    else:
        print(f'El archivo {filename} no contiene -min')

print("Proceso completado.")
