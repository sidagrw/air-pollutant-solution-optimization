import AQISummaryCard from '../components/AQISummaryCard.jsx'
import PollutantChart from '../components/PollutantChart.jsx'
import SimulationPanel from '../components/SimulationPanel.jsx'

export default function Dashboard() {
  return (
    <main className="page">
      <h1>Air Pollutant Solution Optimization</h1>
      <p>Predict AQI, simulate interventions, and compare solution strategies.</p>
      <AQISummaryCard />
      <PollutantChart />
      <SimulationPanel />
    </main>
  )
}
