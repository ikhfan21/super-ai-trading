import subprocess
import os
from datetime import datetime

def run_command(command):
    """Menjalankan perintah terminal dan mengembalikan outputnya."""
    try:
        result = subprocess.run(command, check=True, shell=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error saat menjalankan perintah: {' '.join(command)}")
        print(e.stderr)
        return None

def sync_to_github(commit_message):
    """
    Fungsi utama untuk melakukan sinkronisasi (add, commit, push) ke GitHub.
    """
    print("\n--- MEMULAI SINKRONISASI OTOMATIS KE GITHUB ---")
    
    # 1. Tambahkan semua perubahan
    print("1. Menambahkan semua perubahan (git add .)...")
    run_command("git add .")
    
    # 2. Buat commit dengan pesan yang dinamis
    # Tambahkan stempel waktu ke pesan commit agar unik
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_commit_message = f'"{commit_message} - {timestamp}"'
    print(f"2. Membuat commit dengan pesan: {full_commit_message}...")
    run_command(f"git commit -m {full_commit_message}")
    
    # 3. Kirim ke GitHub
    print("3. Mengirim perubahan ke GitHub (git push)...")
    push_output = run_command("git push origin main")
    
    if push_output is not None:
        print("--- SINKRONISASI BERHASIL ---")
        if "Uploading LFS objects" in push_output:
            print("File besar (LFS) berhasil diunggah.")
    else:
        print("--- SINKRONISASI GAGAL ---")

if __name__ == "__main__":
    # Contoh cara penggunaan jika file ini dijalankan langsung
    print("Ini adalah script utilitas. Jalankan script lain seperti get_data.py untuk menggunakannya.")
    # sync_to_github("Tes sinkronisasi manual")
