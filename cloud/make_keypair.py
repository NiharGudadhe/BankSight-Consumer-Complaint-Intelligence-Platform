from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "secrets"
OUT.mkdir(exist_ok=True)

private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

private_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),   # fine for a local trial; use a passphrase for real use
)
public_pem = private_key.public_key().public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
)

(OUT / "snowflake_key.p8").write_bytes(private_pem)
(OUT / "snowflake_key.pub").write_bytes(public_pem)

body = "".join(public_pem.decode().strip().splitlines()[1:-1])
print("Saved keys in", OUT)
print("\nRun this in Snowflake (one line):\n")
print(f"ALTER USER banksight_svc SET RSA_PUBLIC_KEY = '{body}';")
