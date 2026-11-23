import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import {
  BrainCircuit,
  LayoutDashboard,
  FileText,
  Network,
  MessageSquare,
  GitCompareArrows,
  Settings,
  Search,
  Bell,
  UploadCloud,
  Files,
  Sparkles,
  Database,
  Activity,
  Plus,
  Eye,
  X,
  RefreshCw,
  Trash2,
  Copy,
  Check
} from "lucide-react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  MarkerType
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import "./App.css";
const API_URL = "https://graphrag-project.onrender.com";
const ENTITY_COLORS = {
  Person: "#a78bfa",
  Organization: "#60a5fa",
  Project: "#34d399",
  Department: "#fbbf24",
  Location: "#fb923c",
  Entity: "#94a3b8"
};

function getApiErrorMessage(error, fallback) {
  const detail = error?.response?.data?.detail;

  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (typeof item === "string") return item;
        if (item && typeof item.msg === "string") return item.msg;
        if (item && typeof item.message === "string") return item.message;
        return "";
      })
      .filter(Boolean);

    if (messages.length) return messages.join(" ");
  }

  if (detail && typeof detail === "object") {
    if (typeof detail.message === "string") return detail.message;
    if (typeof detail.msg === "string") return detail.msg;
  }

  return fallback;
}

function SidebarButton({
  icon: Icon,
  children,
  active = false,
  disabled = false,
  onClick
}) {
  return (
    <button
      type="button"
      className={`nav-item ${active ? "active" : ""}`}
      onClick={onClick}
      disabled={disabled}
      aria-current={active ? "page" : undefined}
      title={disabled ? `${children} (not available)` : children}
    >
      <Icon size={18} />
      <span>{children}</span>
    </button>
  );
}
function Stat({ icon, title, value }) {
  const descriptions = {
    Documents: "Successfully uploaded",
    "Graph Entities": "Entities in selected graph",
    Relationships: "Graph connections",
    Queries: "Successful AI queries"
  };
  return (
    <div className="stat-card">
      <div className="stat-icon">{icon}</div>
      <div className="stat-details">
        <span className="stat-label">{title}</span>
        <h2 className="stat-value">{value}</h2>
        <small className="stat-description">
          {descriptions[title] || ""}
        </small>
      </div>
    </div>
  );
}
function App() {
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  const [messageType, setMessageType] = useState("");
  const [deletingDocumentId, setDeletingDocumentId] = useState("");
  const [deleteMessage, setDeleteMessage] = useState("");
  const [deleteMessageType, setDeleteMessageType] = useState("");
  const [copiedDocumentId, setCopiedDocumentId] = useState("");
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [extractedText, setExtractedText] = useState("");
  const [loadingText, setLoadingText] = useState(false);
  const [selectedGraphDocumentId, setSelectedGraphDocumentId] =
    useState("");
  const [selectedAssistantDocumentId, setSelectedAssistantDocumentId] =
    useState("");
  const [graphNodes, setGraphNodes] = useState([]);
  const [graphEdges, setGraphEdges] = useState([]);
  const [loadingGraph, setLoadingGraph] = useState(false);
  const [graphMessage, setGraphMessage] = useState("");
  const [graphError, setGraphError] = useState("");
  const [assistantQuery, setAssistantQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [aiAnswer, setAiAnswer] = useState("");
  const [searching, setSearching] = useState(false);
  const [assistantError, setAssistantError] = useState("");
  const [assistantMessage, setAssistantMessage] = useState("");
  const [queryCount, setQueryCount] = useState(0);
  const [activeSection, setActiveSection] = useState("dashboard");
  const sectionNames = {
    dashboard: "Dashboard",
    documents: "Documents",
    graph: "Knowledge Graph",
    assistant: "AI Assistant"
  };
  // Load documents from the backend.
  const loadDocuments = useCallback(async () => {
    try {
      const response = await axios.get(`${API_URL}/api/documents`);
      const loadedDocuments = response.data.documents || [];
      setDocuments(loadedDocuments);
      setSelectedGraphDocumentId((current) => {
        if (loadedDocuments.some((doc) => doc.id === current)) {
          return current;
        }
        return loadedDocuments[0]?.id || "";
      });
      setSelectedAssistantDocumentId((current) => {
        if (loadedDocuments.some((doc) => doc.id === current)) {
          return current;
        }
        return loadedDocuments[0]?.id || "";
      });
    } catch (error) {
      console.error("Unable to load documents:", error);
    }
  }, []);
  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);
  // Automatically update the active sidebar item while scrolling.
  useEffect(() => {
    const sectionMap = [
      { id: "dashboard-section", key: "dashboard" },
      { id: "documents-section", key: "documents" },
      { id: "graph-section", key: "graph" },
      { id: "assistant-section", key: "assistant" }
    ];
    const observedSections = sectionMap
      .map((section) => ({
        ...section,
        element: document.getElementById(section.id)
      }))
      .filter((section) => section.element);
    if (!observedSections.length) return;
    const observer = new IntersectionObserver(
      (entries) => {
        const visibleSections = entries
          .filter((entry) => entry.isIntersecting)
          .map((entry) => {
            const section = observedSections.find(
              (item) => item.element === entry.target
            );
            return {
              key: section?.key,
              top: entry.boundingClientRect.top
            };
          })
          .filter((item) => item.key);
        if (visibleSections.length > 0) {
          visibleSections.sort(
            (a, b) => Math.abs(a.top) - Math.abs(b.top)
          );
          setActiveSection(visibleSections[0].key);
        }
      },
      {
        root: null,
        rootMargin: "-15% 0px -65% 0px",
        threshold: 0
      }
    );
    observedSections.forEach(({ element }) => {
      observer.observe(element);
    });
    return () => {
      observer.disconnect();
    };
  }, []);
  // Navigate to a section.
  const navigateTo = (sectionId, sectionKey) => {
    setActiveSection(sectionKey);
    const target = document.getElementById(sectionId);
    if (target) {
      target.scrollIntoView({
        behavior: "smooth",
        block: "start"
      });
    }
  };
  // Upload PDF files.
  const handleUpload = async (event) => {
    const input = event.currentTarget;
    const files = Array.from(input.files || []);
    if (!files.length) return;
    setUploading(true);
    setUploadMessage("");
    setMessageType("");
    let successfulUploads = 0;
    try {
      for (const file of files) {
        const formData = new FormData();
        formData.append("file", file);
        const response = await axios.post(
          `${API_URL}/api/documents/upload`,
          formData
        );
        const uploadedDocument = response.data.document;
        setDocuments((previous) => [
          ...previous.filter((doc) => doc.id !== uploadedDocument.id),
          uploadedDocument
        ]);
        setSelectedGraphDocumentId(uploadedDocument.id);
        setSelectedAssistantDocumentId(uploadedDocument.id);
        successfulUploads++;
      }
      setUploadMessage(
        `${successfulUploads} PDF file(s) uploaded and text extracted successfully!`
      );
      setMessageType("success");
    } catch (error) {
      console.error("Upload failed:", error);
      setUploadMessage(
        getApiErrorMessage(
          error,
          "Upload or text extraction failed. Please try again."
        )
      );
      setMessageType("error");
    } finally {
      setUploading(false);
      input.value = "";
    }
  };
  // Delete a document.
  const deleteDocument = async (document) => {
    const confirmed = window.confirm(
      `Are you sure you want to delete "${document.filename}"?\n\nDocument ID: ${document.id}\n\nThis action cannot be undone.`
    );
    if (!confirmed) return;
    setDeletingDocumentId(document.id);
    setDeleteMessage("");
    setDeleteMessageType("");
    try {
      await axios.delete(
        `${API_URL}/api/documents/${document.id}`
      );
      setDocuments((previous) =>
        previous.filter((item) => item.id !== document.id)
      );
      if (selectedGraphDocumentId === document.id) {
        setSelectedGraphDocumentId("");
        setGraphNodes([]);
        setGraphEdges([]);
        setGraphMessage("");
        setGraphError("");
      }
      if (selectedAssistantDocumentId === document.id) {
        setSelectedAssistantDocumentId("");
        setAiAnswer("");
        setSearchResults([]);
        setAssistantError("");
        setAssistantMessage("");
      }
      if (selectedDocument?.id === document.id) {
        setSelectedDocument(null);
        setExtractedText("");
      }
      setDeleteMessage(
        `"${document.filename}" was deleted successfully.`
      );
      setDeleteMessageType("success");
    } catch (error) {
      console.error("Document deletion failed:", error);
      setDeleteMessage(
        getApiErrorMessage(
          error,
          "Unable to delete this document. Please check your backend DELETE endpoint."
        )
      );
      setDeleteMessageType("error");
    } finally {
      setDeletingDocumentId("");
    }
  };
  // Copy document ID.
  const copyDocumentId = async (documentId) => {
    try {
      await navigator.clipboard.writeText(documentId);
      setCopiedDocumentId(documentId);
      window.setTimeout(() => {
        setCopiedDocumentId((current) =>
          current === documentId ? "" : current
        );
      }, 1800);
    } catch (error) {
      console.error("Unable to copy document ID:", error);
      window.prompt("Copy this document ID:", documentId);
    }
  };
  // View extracted PDF text.
  const viewExtractedText = async (document) => {
    setSelectedDocument(document);
    setExtractedText("");
    setLoadingText(true);
    try {
      const response = await axios.get(
        `${API_URL}/api/documents/${document.id}/text`
      );
      setExtractedText(response.data.extracted_text || "");
    } catch (error) {
      console.error("Unable to retrieve extracted text:", error);
      setExtractedText("Unable to retrieve extracted text.");
    } finally {
      setLoadingText(false);
    }
  };
  // Change AI Assistant document.
  const handleAssistantDocumentChange = (documentId) => {
    setSelectedAssistantDocumentId(documentId);
    setSearchResults([]);
    setAiAnswer("");
    setAssistantError("");
    setAssistantMessage("");
  };
  // Ask a question using Gemini.
  const handleSemanticSearch = async (event) => {
    event.preventDefault();
    const question = assistantQuery.trim();
    if (!question) {
      setAssistantError("Please enter a question.");
      return;
    }
    if (!selectedAssistantDocumentId) {
      setAssistantError("Please select a document first.");
      return;
    }
    setSearching(true);
    setAssistantError("");
    setAssistantMessage("");
    setSearchResults([]);
    setAiAnswer("");
    try {
      const response = await axios.post(
        `${API_URL}/api/knowledge/ask`,
        {
          question,
          document_id: selectedAssistantDocumentId,
          top_k: 3
        }
      );
      const answer = response.data.answer || "";
      const results = response.data.sources || [];
      setAiAnswer(answer);
      setSearchResults(results);
      setQueryCount((previous) => previous + 1);
      if (answer) {
        setAssistantMessage(
          `Gemini generated an answer using ${results.length} source chunk(s).`
        );
      } else {
        setAssistantMessage(
          "No answer was generated. Please try another question."
        );
      }
    } catch (error) {
      console.error("Semantic search failed:", error);
      setAssistantError(
        getApiErrorMessage(
          error,
          "Unable to search your document. Please check your backend connection."
        )
      );
    } finally {
      setSearching(false);
    }
  };
  // Load knowledge graph from Neo4j.
  const loadGraph = useCallback(async (documentId) => {
    if (!documentId) {
      setGraphNodes([]);
      setGraphEdges([]);
      setGraphMessage("");
      setGraphError("");
      return;
    }
    setLoadingGraph(true);
    setGraphError("");
    setGraphMessage("");
    try {
      const response = await axios.get(
        `${API_URL}/api/knowledge/graph`,
        {
          params: { document_id: documentId }
        }
      );
      const data = response.data;
      const sourceNodes = data.nodes || [];
      const nodes = sourceNodes.map((node, index) => {
        const column = index % 4;
        const row = Math.floor(index / 4);
        const entityType = node.type || "Entity";
        const entityColor =
          ENTITY_COLORS[entityType] || ENTITY_COLORS.Entity;
        return {
          id: String(node.id),
          position: {
            x: 80 + column * 260,
            y: 70 + row * 155
          },
          data: {
            entityType,
            label: (
              <div className="graph-node-content">
                <strong>{node.name}</strong>
                <span>{entityType}</span>
              </div>
            )
          },
          style: {
            background: "#17152c",
            color: "#ffffff",
            border: `1px solid ${entityColor}`,
            borderRadius: "12px",
            padding: "12px",
            width: 190,
            fontSize: "12px",
            textAlign: "center"
          }
        };
      });
      const edges = (data.edges || data.relationships || []).map(
        (relationship, index) => ({
          id: String(relationship.id || `edge-${index}`),
          source: String(relationship.source),
          target: String(relationship.target),
          label: relationship.type,
          type: "smoothstep",
          animated: true,
          markerEnd: {
            type: MarkerType.ArrowClosed,
            color: "#9b8afb"
          },
          style: {
            stroke: "#9b8afb",
            strokeWidth: 1.8
          },
          labelStyle: {
            fill: "#d8d2ff",
            fontSize: 10,
            fontWeight: 600
          },
          labelBgStyle: {
            fill: "#211d3b",
            fillOpacity: 0.95
          },
          labelBgPadding: [5, 3],
          labelBgBorderRadius: 4
        })
      );
      setGraphNodes(nodes);
      setGraphEdges(edges);
      setGraphMessage(
        `${data.total_nodes ?? nodes.length} entities and ${
          data.total_relationships ?? edges.length
        } relationships loaded from Neo4j.`
      );
    } catch (error) {
      console.error("Graph loading failed:", error);
      setGraphNodes([]);
      setGraphEdges([]);
      setGraphError(
        getApiErrorMessage(
          error,
          "Unable to load the knowledge graph. Check your backend connection."
        )
      );
    } finally {
      setLoadingGraph(false);
    }
  }, []);
  // Reload graph when selected document changes.
  useEffect(() => {
    loadGraph(selectedGraphDocumentId);
  }, [selectedGraphDocumentId, loadGraph]);
  // Build knowledge graph.
  const buildGraph = async () => {
    if (!selectedGraphDocumentId) {
      setGraphError("Please select a document first.");
      return;
    }
    setLoadingGraph(true);
    setGraphError("");
    setGraphMessage("Building knowledge graph...");
    try {
      await axios.post(
        `${API_URL}/api/documents/${selectedGraphDocumentId}/build-graph`
      );
      await loadGraph(selectedGraphDocumentId);
    } catch (error) {
      console.error("Graph creation failed:", error);
      setGraphError(
        getApiErrorMessage(error, "Unable to build the knowledge graph.")
      );
      setLoadingGraph(false);
    }
  };
  const graphEntityTypes = [
    ...new Set(graphNodes.map((node) => node.data.entityType))
  ];
  return (
    <div className="app">
      {/* Sidebar */}
      <aside className="sidebar">
        <div
          className="brand"
          onClick={() =>
            navigateTo("dashboard-section", "dashboard")
          }
          role="button"
          tabIndex={0}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              navigateTo("dashboard-section", "dashboard");
            }
          }}
        >
          <BrainCircuit size={30} />
          <span>
            GraphRAG<span className="accent">.</span>
          </span>
        </div>
        <p className="nav-label">WORKSPACE</p>
        <nav>
          <SidebarButton
            icon={LayoutDashboard}
            active={activeSection === "dashboard"}
            onClick={() =>
              navigateTo("dashboard-section", "dashboard")
            }
          >
            Dashboard
          </SidebarButton>
          <SidebarButton
            icon={Files}
            active={activeSection === "documents"}
            onClick={() =>
              navigateTo("documents-section", "documents")
            }
          >
            Documents
          </SidebarButton>
          <SidebarButton
            icon={Network}
            active={activeSection === "graph"}
            onClick={() =>
              navigateTo("graph-section", "graph")
            }
          >
            Knowledge Graph
          </SidebarButton>
          <SidebarButton
            icon={MessageSquare}
            active={activeSection === "assistant"}
            onClick={() =>
              navigateTo("assistant-section", "assistant")
            }
          >
            AI Assistant
          </SidebarButton>
          <SidebarButton icon={GitCompareArrows} disabled>
            Compare RAG
          </SidebarButton>
        </nav>
        <div className="sidebar-bottom">
          <SidebarButton icon={Settings} disabled>
            Settings
          </SidebarButton>
          <div className="profile">
            <div className="avatar">AI</div>
            <div>
              <strong>My Workspace</strong>
              <small>Personal account</small>
            </div>
          </div>
        </div>
      </aside>
      {/* Main content */}
      <main className="main">
        {/* Top navigation */}
        <header className="topbar">
          <div className="topbar-title">
            <span className="breadcrumb">Workspace</span>
            <span className="breadcrumb-separator">/</span>
            <strong>
              {sectionNames[activeSection] || "Dashboard"}
            </strong>
          </div>
          <div className="top-actions">
            <button
              type="button"
              className="icon-btn"
              aria-label="Search"
              onClick={() =>
                navigateTo("assistant-section", "assistant")
              }
            >
              <Search size={19} />
            </button>
            <button
              type="button"
              className="icon-btn"
              aria-label="Notifications"
              title="Notifications"
            >
              <Bell size={19} />
            </button>
            <div className="avatar">AI</div>
          </div>
        </header>
        <section className="content">
          {/* Welcome section / Dashboard navigation target */}
          <div className="welcome" id="dashboard-section">
            <div className="welcome-copy">
              <span className="eyebrow">
                <Sparkles size={15} />
                INTELLIGENCE WORKSPACE
              </span>
              <h1>Knowledge Explorer</h1>
              <p>
                Transform your documents into connected knowledge.
              </p>
            </div>
            <button
              type="button"
              className="primary-btn"
              onClick={() =>
                navigateTo("upload-section", "dashboard")
              }
            >
              <Plus size={18} />
              New Project
            </button>
          </div>
          {/* Statistics */}
          <div className="stats">
            <Stat
              icon={<Files />}
              title="Documents"
              value={documents.length}
            />
            <Stat
              icon={<Network />}
              title="Graph Entities"
              value={graphNodes.length}
            />
            <Stat
              icon={<Database />}
              title="Relationships"
              value={graphEdges.length}
            />
            <Stat
              icon={<Activity />}
              title="Queries"
              value={queryCount}
            />
          </div>
          {/* Upload section */}
          <div className="section-heading" id="upload-section">
            <div>
              <span className="section-kicker">
                DOCUMENT MANAGEMENT
              </span>
              <h2>Document Intelligence</h2>
              <p>
                Upload PDF files and prepare them for intelligent search.
              </p>
            </div>
          </div>
          <div className="upload-card">
            <div className="upload-icon">
              <UploadCloud size={34} />
            </div>
            <h3>Upload your documents</h3>
            <p>
              Add PDF documents to extract their text, search their content,
              and generate knowledge graphs.
            </p>
            <label className="primary-btn upload-btn">
              <Plus size={18} />
              {uploading ? "Processing PDF..." : "Select PDF files"}
              <input
                type="file"
                accept=".pdf,application/pdf"
                multiple
                hidden
                disabled={uploading}
                onChange={handleUpload}
              />
            </label>
            <small className="upload-hint">
              PDF files supported · Maximum 20 MB per file
            </small>
            {uploadMessage && (
              <p
                className={`upload-message ${messageType}`}
                role="status"
              >
                {uploadMessage}
              </p>
            )}
          </div>
          {/* Uploaded documents */}
          <div
            className="section-heading documents-heading"
            id="documents-section"
          >
            <div>
              <span className="section-kicker">YOUR FILES</span>
              <h2>Uploaded Documents</h2>
              <p>
                View your PDFs, copy document IDs, or delete documents.
              </p>
            </div>
            <span className="document-count">
              {documents.length}{" "}
              {documents.length === 1 ? "document" : "documents"}
            </span>
          </div>
          {deleteMessage && (
            <p
              className={`upload-message ${deleteMessageType}`}
              role={deleteMessageType === "error" ? "alert" : "status"}
            >
              {deleteMessage}
            </p>
          )}
          {documents.length === 0 ? (
            <div className="panel empty-documents">
              <Files size={38} />
              <strong>No documents uploaded yet</strong>
              <span>Upload a PDF to get started.</span>
            </div>
          ) : (
            <div className="panel document-list">
              {documents.map((document) => (
                <div key={document.id} className="document-row">
                  <div className="document-info">
                    <div className="document-file-icon">
                      <FileText size={22} />
                    </div>
                    <div className="document-text">
                      <strong title={document.filename}>
                        {document.filename}
                      </strong>
                      <small>
                        {document.total_pages ?? 0} pages
                        <span className="metadata-dot">•</span>
                        {document.total_characters ?? 0} characters extracted
                      </small>
                      <div className="document-id-row">
                        <span className="document-id-label">
                          Document ID:
                        </span>
                        <code title={document.id}>
                          {document.id}
                        </code>
                        <button
                          type="button"
                          className="copy-id-btn"
                          onClick={() => copyDocumentId(document.id)}
                          title="Copy document ID"
                          aria-label="Copy document ID"
                        >
                          {copiedDocumentId === document.id ? (
                            <Check size={14} />
                          ) : (
                            <Copy size={14} />
                          )}
                          <span>
                            {copiedDocumentId === document.id
                              ? "Copied"
                              : "Copy ID"}
                          </span>
                        </button>
                      </div>
                    </div>
                  </div>
                  <div className="document-actions">
                    <button
                      type="button"
                      className="secondary-btn"
                      onClick={() => viewExtractedText(document)}
                    >
                      <Eye size={16} />
                      View Text
                    </button>
                    <button
                      type="button"
                      className="delete-btn"
                      onClick={() => deleteDocument(document)}
                      disabled={deletingDocumentId === document.id}
                      title="Delete this document"
                    >
                      {deletingDocumentId === document.id ? (
                        <RefreshCw
                          size={16}
                          className="spin-icon"
                        />
                      ) : (
                        <Trash2 size={16} />
                      )}
                      {deletingDocumentId === document.id
                        ? "Deleting..."
                        : "Delete"}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
          {/* Graph and AI sections */}
          <div className="feature-section">
            <div className="section-heading feature-heading">
              <div>
                <span className="section-kicker">
                  EXPLORE YOUR KNOWLEDGE
                </span>
                <h2>Knowledge Graph & AI Assistant</h2>
                <p>
                  Explore document relationships and ask questions about your PDFs.
                </p>
              </div>
            </div>
            <div className="bottom-grid">
              {/* Knowledge Graph */}
              <div className="panel graph-panel" id="graph-section">
                <div className="panel-heading">
                  <div>
                    <span className="panel-kicker">
                      VISUAL EXPLORATION
                    </span>
                    <h3>Knowledge Graph</h3>
                    <p>
                      Explore connected entities and relationships from your document.
                    </p>
                  </div>
                  <div className="panel-heading-icon">
                    <Network size={21} />
                  </div>
                </div>
                <div className="graph-toolbar">
                  <div className="field-group graph-select-group">
                    <label htmlFor="graph-document">
                      Select graph document
                    </label>
                    <select
                      id="graph-document"
                      value={selectedGraphDocumentId}
                      onChange={(event) =>
                        setSelectedGraphDocumentId(event.target.value)
                      }
                      disabled={documents.length === 0}
                    >
                      <option value="">
                        Select a document for the graph
                      </option>
                      {documents.map((document) => (
                        <option key={document.id} value={document.id}>
                          {document.filename}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="graph-actions">
                    <button
                      type="button"
                      className="secondary-btn"
                      onClick={buildGraph}
                      disabled={!selectedGraphDocumentId || loadingGraph}
                    >
                      {loadingGraph ? (
                        <RefreshCw
                          size={16}
                          className="spin-icon"
                        />
                      ) : (
                        <Network size={16} />
                      )}
                      {loadingGraph ? "Building..." : "Build Graph"}
                    </button>
                    <button
                      type="button"
                      className="secondary-btn refresh-btn"
                      onClick={() => loadGraph(selectedGraphDocumentId)}
                      disabled={!selectedGraphDocumentId || loadingGraph}
                      title="Refresh graph"
                    >
                      <RefreshCw size={16} />
                      Refresh
                    </button>
                  </div>
                </div>
                {graphMessage && (
                  <p className="graph-status" role="status">
                    <Activity size={15} />
                    {graphMessage}
                  </p>
                )}
                {graphError && (
                  <p className="graph-error" role="alert">
                    {graphError}
                  </p>
                )}
                {graphNodes.length > 0 ? (
                  <div className="graph-visualization">
                    <ReactFlow
                      nodes={graphNodes}
                      edges={graphEdges}
                      fitView
                      fitViewOptions={{ padding: 0.18 }}
                      minZoom={0.1}
                      maxZoom={2}
                      nodesDraggable
                      nodesConnectable={false}
                      elementsSelectable
                      proOptions={{ hideAttribution: true }}
                    >
                      <Background color="#34304f" gap={22} />
                      <Controls />
                      <MiniMap
                        nodeColor={(node) =>
                          ENTITY_COLORS[node.data?.entityType] ||
                          ENTITY_COLORS.Entity
                        }
                        maskColor="rgba(10, 8, 25, 0.7)"
                      />
                    </ReactFlow>
                  </div>
                ) : (
                  <div className="empty-graph graph-empty-state">
                    {loadingGraph ? (
                      <RefreshCw
                        size={40}
                        className="spin-icon"
                      />
                    ) : (
                      <Network size={48} />
                    )}
                    <strong>
                      {loadingGraph
                        ? "Loading knowledge graph..."
                        : "No knowledge graph to display"}
                    </strong>
                    <span>
                      {documents.length === 0
                        ? "Upload a PDF first."
                        : "Select a document and click Build Graph."}
                    </span>
                  </div>
                )}
                {graphNodes.length > 0 && (
                  <div className="graph-legend">
                    <strong>Entity types</strong>
                    {graphEntityTypes.map((type) => (
                      <div className="legend-item" key={type}>
                        <span
                          className="legend-dot"
                          style={{
                            background:
                              ENTITY_COLORS[type] || ENTITY_COLORS.Entity
                          }}
                        />
                        {type}
                      </div>
                    ))}
                  </div>
                )}
              </div>
              {/* AI Assistant */}
              <div className="panel assistant-panel" id="assistant-section">
                <div className="panel-heading">
                  <div>
                    <span className="panel-kicker">
                      AI-POWERED SEARCH
                    </span>
                    <h3>AI Assistant</h3>
                    <p>
                      Ask questions and get answers from your selected PDF.
                    </p>
                  </div>
                  <div className="panel-heading-icon">
                    <BrainCircuit size={21} />
                  </div>
                </div>
                <div className="assistant-form-area">
                  <div className="field-group">
                    <label htmlFor="assistant-document">
                      Select document to search
                    </label>
                    <select
                      id="assistant-document"
                      value={selectedAssistantDocumentId}
                      onChange={(event) =>
                        handleAssistantDocumentChange(event.target.value)
                      }
                      disabled={documents.length === 0 || searching}
                    >
                      <option value="">
                        Select a document to ask questions about
                      </option>
                      {documents.map((document) => (
                        <option key={document.id} value={document.id}>
                          {document.filename}
                        </option>
                      ))}
                    </select>
                  </div>
                  <form
                    onSubmit={handleSemanticSearch}
                    className="assistant-search-form"
                  >
                    <div className="field-group">
                      <label htmlFor="assistant-question">
                        Your question
                      </label>
                      <textarea
                        id="assistant-question"
                        value={assistantQuery}
                        onChange={(event) =>
                          setAssistantQuery(event.target.value)
                        }
                        placeholder="Example: Who manages the RiverWatch project?"
                        rows={4}
                        disabled={searching}
                      />
                    </div>
                    <button
                      type="submit"
                      className="primary-btn ask-btn"
                      disabled={
                        searching ||
                        !assistantQuery.trim() ||
                        !selectedAssistantDocumentId
                      }
                    >
                      {searching ? (
                        <>
                          <RefreshCw
                            size={17}
                            className="spin-icon"
                          />
                          Asking Gemini...
                        </>
                      ) : (
                        <>
                          <Sparkles size={17} />
                          Ask Gemini
                        </>
                      )}
                    </button>
                  </form>
                </div>
                {assistantError && (
                  <p className="assistant-error" role="alert">
                    {assistantError}
                  </p>
                )}
                {assistantMessage && (
                  <p className="assistant-status" role="status">
                    <Sparkles size={15} />
                    {assistantMessage}
                  </p>
                )}
                {aiAnswer && (
                  <div className="answer-card">
                    <div className="answer-heading">
                      <div className="answer-icon">
                        <Sparkles size={17} />
                      </div>
                      <h4>Gemini AI Answer</h4>
                    </div>
                    <p className="answer-text">{aiAnswer}</p>
                  </div>
                )}
                {searchResults.length > 0 && (
                  <div className="sources-section">
                    <div className="sources-heading">
                      <div>
                        <h4>Supporting Document Sources</h4>
                        <p>
                          Relevant text retrieved from your selected PDF.
                        </p>
                      </div>
                      <span className="source-count">
                        {searchResults.length} sources
                      </span>
                    </div>
                    <div className="source-list">
                      {searchResults.map((result, index) => (
                        <div
                          key={result.chunk_id || index}
                          className="source-card"
                        >
                          <div className="source-card-heading">
                            <strong>Source {index + 1}</strong>
                            <span>
                              Chunk {result.chunk_index ?? "N/A"}
                            </span>
                          </div>
                          <p>{result.text}</p>
                          {typeof result.distance === "number" && (
                            <small>
                              Search distance: {result.distance.toFixed(4)}
                            </small>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {!searching &&
                  !assistantError &&
                  !assistantMessage &&
                  !aiAnswer &&
                  searchResults.length === 0 && (
                    <div className="assistant-box">
                      <div className="assistant-empty-icon">
                        <Sparkles size={25} />
                      </div>
                      <strong>Ask your documents with Gemini</strong>
                      <span>
                        Select a PDF, enter a question, and Gemini will answer
                        using relevant information from that document.
                      </span>
                    </div>
                  )}
              </div>
            </div>
          </div>
        </section>
      </main>
      {/* Extracted text modal */}
      {selectedDocument && (
        <div
          className="modal-overlay"
          onClick={(event) => {
            if (event.target === event.currentTarget) {
              setSelectedDocument(null);
            }
          }}
        >
          <div
            className="text-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="extracted-text-title"
          >
            <div className="modal-header">
              <div>
                <span className="section-kicker">
                  DOCUMENT PREVIEW
                </span>
                <h2 id="extracted-text-title">
                  Extracted PDF Text
                </h2>
                <small title={selectedDocument.filename}>
                  {selectedDocument.filename}
                </small>
              </div>
              <button
                type="button"
                className="icon-btn modal-close"
                onClick={() => setSelectedDocument(null)}
                aria-label="Close extracted text"
              >
                <X />
              </button>
            </div>
            <div className="extracted-content">
              {loadingText ? (
                <div className="modal-loading">
                  <RefreshCw
                    size={22}
                    className="spin-icon"
                  />
                  <p>Loading extracted text...</p>
                </div>
              ) : (
                <pre>{extractedText}</pre>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
export default App;