# Kira — Containerization Setup

## Cara pakai

Development lokal (gratis, tanpa GPU AMD):
```
docker compose up --build
```

Build image untuk submission final (sesuai requirement panitia: semua submission wajib di-containerize):
```
docker build -t kira:submission .
```

## Strategi supaya $0 dari kantong sendiri

Container di repo ini SENGAJA tidak menjalankan model 70B di dalamnya.
Dia cuma berisi logic orchestrator + agents, dan memanggil model besar
lewat HTTP ke endpoint eksternal yang diatur di `.env`.

Alurnya:

| Fase | Endpoint reasoning | Biaya |
|---|---|---|
| Hari 1-4 (development & testing logic agen) | Groq free tier / Ollama lokal | Rp 0 |
| Hari 5 (demo + submission final) | MI300X Droplet (AMD Developer Cloud) | Dari kredit $100-150 gratis |

Jangan nyalakan Droplet MI300X lebih awal dari yang perlu. Nyalakan,
test, rekam demo, matikan. Itu pola paling aman supaya kredit gratis
cukup dan tidak pernah menyentuh kartu pembayaran yang terdaftar.

## Checklist sebelum submission

- [ ] `.env` terisi dengan endpoint MI300X final (bukan endpoint dev)
- [ ] `docker build` berhasil tanpa error dari kondisi bersih (clone ulang repo, coba build)
- [ ] Droplet MI300X dalam keadaan menyala saat submission dicek panitia (kalau perlu live demo)
- [ ] Tidak ada API key atau secret ter-commit di git (cek `.dockerignore` dan `.gitignore`)
