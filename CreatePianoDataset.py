import os
import shutil

# Căile directoare pentru fișierele sursă și destinația unde vor fi copiate
audio_folder = "C:\\Users\\pirla\\Downloads\\SoundsWAV\\nsynth-train\\audio"
output_folder = "dataset_keyboard"

# Crează directorul de ieșire dacă nu există
os.makedirs(output_folder, exist_ok=True)


# Funcție pentru a transforma pitch-ul MIDI într-un nume de notă în română
def midi_to_note_name_ro(pitch):
    note_names_ro = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]
    note = note_names_ro[pitch % 12]  # Determină nota pe baza pitch-ului (valoarea MIDI)
    octave = (pitch // 12) - 1  # Calculează octava în care se află nota
    return f"{note}{octave}"  # Returnează nota în formatul românesc (ex: Do3, Re4)


# Parcurgem fișierele din folderul sursă
for file_name in os.listdir(audio_folder):
    if file_name.endswith(".wav"):  # Verificăm doar fișierele .wav
        parts = file_name.split("_")  # Împărțim numele fișierului pe baza caracterului '_'

        if len(parts) > 2:  # Verificăm dacă formatul fișierului este corect
            instrument = parts[0] + "_" + parts[1]  # Extragerea tipului de instrument (ex: keyboard_acoustic)

            # Verificăm dacă instrumentul este de tip "keyboard"
            if "keyboard" in instrument:  # Verificăm dacă instrumentul conține "keyboard"
                # Extragem pitch-ul (al doilea număr după '_')
                try:
                    pitch = int(parts[2].split("-")[1])  # Pitch-ul este al doilea număr din denumirea fișierului
                except ValueError:
                    print(f"Error parsing pitch from {file_name}")
                    continue  # Dacă nu putem extrage pitch-ul, sărim la următorul fișier

                # Transformăm pitch-ul în nota+octava
                note_name = midi_to_note_name_ro(pitch)

                # Creăm folderul pentru nota+octava (ex: Do3) dacă nu există deja
                note_folder = os.path.join(output_folder, note_name)
                os.makedirs(note_folder, exist_ok=True)

                # Copiem fișierul în folderul corespunzător
                source_file = os.path.join(audio_folder, file_name)
                target_file = os.path.join(note_folder, file_name)

                shutil.copy(source_file, target_file)  # Copiem fișierul
                print(f"Copied {file_name} -> {note_name}")  # Afișăm un mesaj de confirmare
            else:
                print(f"Skipped (not keyboard): {file_name}")  # Dacă nu este un instrument de tip "keyboard", îl sărim

        else:
             print(f"Filename format not recognized: {file_name}")  # Dacă numele fișierului nu se potrivește, îl sărim
    else:
        print(f"Skipped (not piano): {file_name}")  # Dacă fișierul nu este un .wav, îl sărim

print("✅ Toate fișierele au fost procesate!")