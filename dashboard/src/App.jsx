import { useState, useEffect } from 'react';
import { Brain, Memory, Network, Activity, Plus, Send, Trash2, RefreshCw } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

function App() {
  const [activeTab, setActiveTab] = useState('agent');
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [memoryStats, setMemoryStats] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [networkStatus, setNetworkStatus] = useState(null);
  const [models, setModels] = useState([]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    try {
      const [metricsRes, networkRes, modelsRes] = await Promise.all([
        fetch(`${API_URL}/metrics`).catch(() => null),
        fetch(`${API_URL}/network/status`).catch(() => null),
        fetch(`${API_URL}/models`).catch(() => null),
      ]);

      if (metricsRes?.ok) {
        const data = await metricsRes.json();
        setMetrics(data);
      }

      if (networkRes?.ok) {
        const data = await networkRes.json();
        setNetworkStatus(data);
      }

      if (modelsRes?.ok) {
        const data = await modelsRes.json();
        setModels(data.models || []);
      }
    } catch (e) {
      console.error('Failed to fetch data:', e);
    }
  };

  const runAgent = async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/agent/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });
      const data = await res.json();
      setResponse(data.content || data.error || 'No response');
    } catch (e) {
      setResponse('Error: ' + e.message);
    }
    setLoading(false);
  };

  const clearMemory = async () => {
    await fetch(`${API_URL}/memory/clear`, { method: 'POST' });
    fetchData();
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Brain className="w-8 h-8 text-blue-500" />
            <h1 className="text-xl font-bold">NeuronMesh Dashboard</h1>
          </div>
          <button
            onClick={fetchData}
            className="p-2 hover:bg-gray-700 rounded-lg"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
        </div>
      </header>

      {/* Stats Bar */}
      <div className="grid grid-cols-4 gap-4 p-4 bg-gray-800 border-b border-gray-700">
        <StatCard
          icon={<Brain className="w-5 h-5 text-blue-500" />}
          label="Requests"
          value={metrics?.requests?.total || 0}
        />
        <StatCard
          icon={<Activity className="w-5 h-5 text-green-500" />}
          label="P95 Latency"
          value={`${metrics?.latency_ms?.p95?.toFixed(0) || 0}ms`}
        />
        <StatCard
          icon={<Memory className="w-5 h-5 text-purple-500" />}
          label="Total Cost"
          value={`$${metrics?.cost?.total?.toFixed(4) || 0}`}
        />
        <StatCard
          icon={<Network className="w-5 h-5 text-orange-500" />}
          label="Nodes"
          value={networkStatus?.stats?.total_nodes || 0}
        />
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-700">
        {[
          { id: 'agent', icon: Brain, label: 'Agent' },
          { id: 'memory', icon: Memory, label: 'Memory' },
          { id: 'network', icon: Network, label: 'Network' },
          { id: 'models', icon: Activity, label: 'Models' },
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 px-6 py-3 border-b-2 transition-colors ${
              activeTab === tab.id
                ? 'border-blue-500 text-blue-500'
                : 'border-transparent hover:text-gray-300'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="p-6">
        {activeTab === 'agent' && (
          <div className="max-w-2xl">
            <h2 className="text-lg font-semibold mb-4">Run Agent</h2>
            <div className="space-y-4">
              <textarea
                value={prompt}
                onChange={e => setPrompt(e.target.value)}
                placeholder="Enter your prompt..."
                className="w-full h-32 bg-gray-800 border border-gray-700 rounded-lg p-4 text-white resize-none focus:outline-none focus:border-blue-500"
              />
              <button
                onClick={runAgent}
                disabled={loading}
                className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg disabled:opacity-50"
              >
                <Send className="w-4 h-4" />
                {loading ? 'Running...' : 'Run'}
              </button>
              {response && (
                <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 whitespace-pre-wrap">
                  {response}
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'memory' && (
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <h2 className="text-lg font-semibold">Memory Statistics</h2>
              <button
                onClick={clearMemory}
                className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg"
              >
                <Trash2 className="w-4 h-4" />
                Clear All
              </button>
            </div>
            <div className="grid grid-cols-4 gap-4">
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="text-2xl font-bold">{metrics?.requests?.total || 0}</div>
                <div className="text-gray-400">Total Memories</div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'network' && (
          <div className="space-y-4">
            <h2 className="text-lg font-semibold">Network Status</h2>
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="text-2xl font-bold">{networkStatus?.stats?.total_nodes || 0}</div>
                <div className="text-gray-400">Total Nodes</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="text-2xl font-bold">{networkStatus?.stats?.online_nodes || 0}</div>
                <div className="text-gray-400">Online</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="text-2xl font-bold">{networkStatus?.stats?.gpu_nodes || 0}</div>
                <div className="text-gray-400">GPU Nodes</div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'models' && (
          <div>
            <h2 className="text-lg font-semibold mb-4">Available Models</h2>
            <div className="grid grid-cols-2 gap-4">
              {models.map(model => (
                <div key={model.name} className="bg-gray-800 rounded-lg p-4">
                  <div className="font-medium">{model.name}</div>
                  <div className="text-sm text-gray-400">{model.provider}</div>
                  <div className="text-sm text-blue-400">
                    {model.cost_per_1k > 0 ? `$${model.cost_per_1k.toFixed(4)}/1K` : 'FREE'}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ icon, label, value }) {
  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex items-center gap-2 text-gray-400 mb-1">
        {icon}
        <span className="text-sm">{label}</span>
      </div>
      <div className="text-2xl font-bold">{value}</div>
    </div>
  );
}

export default App;
