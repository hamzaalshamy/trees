import { C } from "./theme";
import GlobalHeader from "./GlobalHeader";

const PARAGRAPHS = [
  "I've always found that building visualizations helps me truly understand a concept. Breaking a model down to its components — watching it select features, compute impurity, partition data — makes the learning stick in a way that reading equations or documentation alone never does.",
  "I've been doing this for a while for my own learning. With increasingly capable tools like Claude Code, it's become much easier to extract the visuals I have in my mind into working code in a short period of time.",
  "I thought it was worth sharing these visualizations for anyone learning about tree-based methods. The goal is simple: let you actually see how these models are built, step by step, and organize the relationships between different tree-based methods so the whole family of algorithms clicks together.",
  "Trees is an ongoing project. More algorithms and features are on the way.",
];

export default function About() {
  return (
    <div
      style={{
        minHeight: "100vh",
        background: C.bg,
        color: C.text,
        display: "flex",
        flexDirection: "column",
      }}
    >
      <GlobalHeader />

      <div
        style={{
          maxWidth: 600,
          width: "100%",
          margin: "0 auto",
          padding: "64px 24px 80px",
        }}
      >
        <h1
          style={{
            fontSize: 26,
            fontWeight: 800,
            margin: "0 0 40px",
            background: `linear-gradient(135deg, ${C.accent}, ${C.green})`,
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            backgroundClip: "text",
            letterSpacing: "-0.5px",
          }}
        >
          About Trees
        </h1>

        <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>
          {PARAGRAPHS.map((p, i) => (
            <p
              key={i}
              style={{
                margin: 0,
                fontSize: 13,
                lineHeight: 1.85,
                color: i === PARAGRAPHS.length - 1 ? C.dim : C.text,
              }}
            >
              {p}
            </p>
          ))}
        </div>

        <div
          style={{
            marginTop: 52,
            paddingTop: 24,
            borderTop: `1px solid ${C.border}`,
            display: "flex",
            flexDirection: "column",
            gap: 10,
          }}
        >
          <LinkRow
            label="Built by Hamza Alshamy"
            href="https://hamzaalshamy.github.io"
          />
          <LinkRow
            label="github.com/hamzaalshamy/trees"
            href="https://github.com/hamzaalshamy/trees"
          />
        </div>
      </div>
    </div>
  );
}

function LinkRow({ label, href }) {
  return (
    <div style={{ fontSize: 11, color: C.dimmer }}>
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        style={{
          color: C.dim,
          textDecoration: "none",
          borderBottom: `1px solid ${C.border}`,
          paddingBottom: 1,
          transition: "color 0.15s, border-color 0.15s",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.color = C.text;
          e.currentTarget.style.borderBottomColor = C.dim;
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.color = C.dim;
          e.currentTarget.style.borderBottomColor = C.border;
        }}
      >
        {label}
      </a>
    </div>
  );
}
