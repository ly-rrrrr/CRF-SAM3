import { useMemo, useState } from "react";

const DATASET_OPTIONS = [
  { value: "all", label: "All datasets" },
  { value: "davis17", label: "DAVIS17" },
  { value: "mevis", label: "MeViS" },
  { value: "refytros", label: "Ref-YTROS / ROS-21" },
  { value: "refytros21", label: "Ref-YTROS-21 only" },
];

const SPLIT_OPTIONS = [
  { value: "all", label: "All splits" },
  { value: "train", label: "train" },
  { value: "valid", label: "valid" },
  { value: "valid_u", label: "valid_u" },
];

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return response.json();
}

function useQueryUrl(videoId, dataset, split) {
  return useMemo(() => {
    const params = new URLSearchParams();
    if (videoId) params.set("video_id", videoId);
    params.set("dataset", dataset);
    params.set("split", split);
    params.set("mode", "contains");
    return `/api/search?${params.toString()}`;
  }, [dataset, split, videoId]);
}

function SearchForm({ videoId, setVideoId, dataset, setDataset, split, setSplit, onSearch, loading }) {
  return (
    <form className="search-panel" onSubmit={onSearch}>
      <div className="field">
        <label htmlFor="video-id">Video ID</label>
        <input
          id="video-id"
          value={videoId}
          onChange={(event) => setVideoId(event.target.value)}
          placeholder="e.g. 05d7715782"
        />
      </div>

      <div className="field">
        <label htmlFor="dataset">Dataset</label>
        <select id="dataset" value={dataset} onChange={(event) => setDataset(event.target.value)}>
          {DATASET_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <div className="field">
        <label htmlFor="split">Split</label>
        <select id="split" value={split} onChange={(event) => setSplit(event.target.value)}>
          {SPLIT_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <button className="primary-button" type="submit" disabled={loading || !videoId.trim()}>
        {loading ? "Searching..." : "Search"}
      </button>
    </form>
  );
}

function downloadUrl(match) {
  const params = new URLSearchParams({
    dataset: match.dataset,
    split: match.split,
    video_id: match.video_id,
  });
  return `/download?${params.toString()}`;
}

function formatExpLabel(expId) {
  return expId === "__video__" ? "video masks" : `exp ${expId}`;
}

function MatchCard({ match, onOpen }) {
  return (
    <article className="match-card">
      <img className="match-preview" src={match.preview_url} alt={`${match.video_id} preview`} loading="lazy" />
      <div className="match-meta">
        <div className="match-title-row">
          <h3>{match.video_id}</h3>
          <span className="badge">{match.dataset}</span>
          <span className="badge muted">{match.split}</span>
        </div>
        <p>{match.frame_count} frames</p>
        <code>{match.video_dir}</code>
        <div className="action-row">
          <button className="secondary-button" type="button" onClick={() => onOpen(match)}>
            View All Frames
          </button>
          <a className="download-link" href={downloadUrl(match)}>
            Download ZIP
          </a>
        </div>
      </div>
    </article>
  );
}

function MaskPanel({ maskInfo, selectedExpId, setSelectedExpId, maskGallery, onLoadMaskGallery, onSelectFrame }) {
  if (!maskInfo) return null;

  return (
    <section className="gallery-section">
      <div className="section-header">
        <div>
          <h2>Masks</h2>
          <p>{maskInfo.exp_ids?.length || 0} expression sequence(s)</p>
        </div>
      </div>

      {maskInfo.exp_ids?.length ? (
        <>
          <div className="exp-chip-row">
            {maskInfo.exp_ids.map((entry) => (
              <button
                key={entry.exp_id}
                type="button"
                className={`exp-chip ${selectedExpId === entry.exp_id ? "active" : ""}`}
                onClick={() => {
                  setSelectedExpId(entry.exp_id);
                  onLoadMaskGallery(entry.exp_id);
                }}
              >
                {formatExpLabel(entry.exp_id)} ({entry.frame_count})
              </button>
            ))}
          </div>

          {maskGallery ? (
            <div className="mask-gallery">
              <div className="gallery-header">
                <div>
                  <h3>Mask Sequence: {formatExpLabel(maskGallery.exp_id)}</h3>
                  <p>{maskGallery.frame_count} mask frames</p>
                </div>
                <code>{maskGallery.mask_dir}</code>
              </div>

              <div className="frame-grid">
                {maskGallery.frames.map((frame) => (
                  <button
                    key={`${maskGallery.exp_id}-${frame.name}`}
                    className="frame-card"
                    type="button"
                    onClick={() => onSelectFrame(frame)}
                    title={frame.name}
                  >
                    <img src={frame.url} alt={`mask ${frame.name}`} loading="lazy" />
                    <span>{frame.name}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : null}
        </>
      ) : (
        <div className="empty-note">No expression-level masks were found for this video in the selected dataset split.</div>
      )}
    </section>
  );
}

function FrameGrid({ gallery, onSelectFrame }) {
  if (!gallery) return null;

  return (
    <section className="gallery-section">
      <div className="gallery-header">
        <div>
          <h2>
            {gallery.video_id} <span className="badge">{gallery.dataset}</span> <span className="badge muted">{gallery.split}</span>
          </h2>
          <p>{gallery.frame_count} frames</p>
        </div>
        <div className="gallery-side">
          <a
            className="download-link"
            href={downloadUrl({ dataset: gallery.dataset, split: gallery.split, video_id: gallery.video_id })}
          >
            Download ZIP
          </a>
          <code>{gallery.video_dir}</code>
        </div>
      </div>

      <div className="frame-grid">
        {gallery.frames.map((frame) => (
          <button
            key={frame.name}
            className="frame-card"
            type="button"
            onClick={() => onSelectFrame(frame)}
            title={frame.name}
          >
            <img src={frame.url} alt={frame.name} loading="lazy" />
            <span>{frame.name}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

function Lightbox({ frame, onClose }) {
  if (!frame) return null;

  return (
    <div className="lightbox" onClick={onClose}>
      <div className="lightbox-panel" onClick={(event) => event.stopPropagation()}>
        <div className="lightbox-header">
          <strong>{frame.name}</strong>
          <button className="secondary-button" type="button" onClick={onClose}>
            Close
          </button>
        </div>
        <img className="lightbox-image" src={frame.url} alt={frame.name} />
      </div>
    </div>
  );
}

export default function App() {
  const [videoId, setVideoId] = useState("05d7715782");
  const [dataset, setDataset] = useState("all");
  const [split, setSplit] = useState("all");
  const [loading, setLoading] = useState(false);
  const [matches, setMatches] = useState([]);
  const [error, setError] = useState("");
  const [gallery, setGallery] = useState(null);
  const [maskInfo, setMaskInfo] = useState(null);
  const [maskGallery, setMaskGallery] = useState(null);
  const [selectedExpId, setSelectedExpId] = useState("");
  const [selectedFrame, setSelectedFrame] = useState(null);

  const searchUrl = useQueryUrl(videoId.trim(), dataset, split);

  async function handleSearch(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setGallery(null);
    setMaskInfo(null);
    setMaskGallery(null);
    setSelectedExpId("");
    setSelectedFrame(null);

    try {
      const payload = await fetchJson(searchUrl);
      setMatches(payload.matches || []);
      if (!payload.matches || payload.matches.length === 0) {
        setError("No matching frame directories were found.");
      }
    } catch (err) {
      setMatches([]);
      setError(err.message || "Search failed.");
    } finally {
      setLoading(false);
    }
  }

  async function openGallery(match) {
    setLoading(true);
    setError("");
    setSelectedFrame(null);
    try {
      const params = new URLSearchParams({
        dataset: match.dataset,
        split: match.split,
        video_id: match.video_id,
      });
      const payload = await fetchJson(`/api/frames?${params.toString()}`);
      setGallery(payload);

      const maskPayload = await fetchJson(`/api/masks?${params.toString()}`);
      setMaskInfo(maskPayload);
      if (maskPayload.exp_ids && maskPayload.exp_ids.length > 0) {
        const firstExpId = maskPayload.exp_ids[0].exp_id;
        setSelectedExpId(firstExpId);
        const maskParams = new URLSearchParams({
          ...Object.fromEntries(params.entries()),
          exp_id: firstExpId,
        });
        const firstMaskGallery = await fetchJson(`/api/mask-frames?${maskParams.toString()}`);
        setMaskGallery(firstMaskGallery);
      } else {
        setMaskGallery(null);
      }
    } catch (err) {
      setGallery(null);
      setMaskInfo(null);
      setMaskGallery(null);
      setError(err.message || "Failed to load frames.");
    } finally {
      setLoading(false);
    }
  }

  async function loadMaskGallery(expId) {
    if (!gallery) return;
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams({
        dataset: gallery.dataset,
        split: gallery.split,
        video_id: gallery.video_id,
        exp_id: expId,
      });
      const payload = await fetchJson(`/api/mask-frames?${params.toString()}`);
      setMaskGallery(payload);
    } catch (err) {
      setMaskGallery(null);
      setError(err.message || "Failed to load masks.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Local Dataset Browser</p>
          <h1>Frame Gallery Agent</h1>
          <p className="hero-copy">
            Query a <code>video_id</code>, inspect the matching raw frame directory, and browse every frame in the browser.
          </p>
        </div>
      </header>

      <SearchForm
        videoId={videoId}
        setVideoId={setVideoId}
        dataset={dataset}
        setDataset={setDataset}
        split={split}
        setSplit={setSplit}
        onSearch={handleSearch}
        loading={loading}
      />

      {error ? <div className="error-banner">{error}</div> : null}

      {matches.length > 0 ? (
        <section className="matches-section">
          <div className="section-header">
            <h2>Matches</h2>
            <p>{matches.length} result(s)</p>
          </div>
          <div className="matches-grid">
            {matches.map((match) => (
              <MatchCard
                key={`${match.dataset}-${match.split}-${match.video_id}`}
                match={match}
                onOpen={openGallery}
              />
            ))}
          </div>
        </section>
      ) : null}

      <FrameGrid gallery={gallery} onSelectFrame={setSelectedFrame} />
      <MaskPanel
        maskInfo={maskInfo}
        selectedExpId={selectedExpId}
        setSelectedExpId={setSelectedExpId}
        maskGallery={maskGallery}
        onLoadMaskGallery={loadMaskGallery}
        onSelectFrame={setSelectedFrame}
      />
      <Lightbox frame={selectedFrame} onClose={() => setSelectedFrame(null)} />
    </main>
  );
}
