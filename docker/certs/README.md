# Extra CA certificates

Leave this folder empty unless your network intercepts HTTPS (for example, Zscaler or a corporate proxy). In that case `docker build` fails with errors like `unable to get local issuer certificate`.

If it does, export your network's root CA as a PEM file with a `.crt` extension into this folder, then rebuild. The image will trust it.

On macOS, if the CA is in the System keychain (replace `Zscaler` with your CA's name):

```bash
security find-certificate -a -c "Zscaler" -p /Library/Keychains/System.keychain > docker/certs/network-root.crt
```

On Linux, your IT department can usually provide the file, or it is already in `/usr/local/share/ca-certificates/`.

`*.crt` files in this folder are gitignored.
