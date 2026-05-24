# Deploying the demo to GitHub Pages

The site lives in this `docs/` folder and is served as a static GitHub Pages
site straight from the `main` branch.

## One-time setup

1. Push this branch to GitHub (already done if you can see `docs/` on github.com).
2. Go to **Settings → Pages** on the repository.
3. Under **Build and deployment → Source**, choose **Deploy from a branch**.
4. Set **Branch** to `main` and the folder to **`/docs`**, then **Save**.
5. Wait ~1 minute. Your site goes live at:

   ```
   https://riptcodes.github.io/Student-Engagement-Analysis-Using-Facial-Recognition-ResNet50V2/
   ```

GitHub Pages is served over HTTPS, which the browser requires before it will
grant webcam access — so the camera works on the live URL.

## Updating the site

Any push to `main` that touches `docs/` triggers an automatic redeploy.

## Add the model

The live scoring needs the converted model in `docs/model/`. See
[`CONVERT_MODEL.md`](CONVERT_MODEL.md).

## Testing locally

Open a terminal in the repo root and serve the folder (a plain `file://` open
will block the webcam and model fetch):

```bash
python -m http.server -d docs 8000
```

Then visit <http://localhost:8000/>. `localhost` counts as a secure context, so
the camera works without HTTPS.
