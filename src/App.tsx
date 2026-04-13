import Hero from './components/Hero'
import StatsBar from './components/StatsBar'
import ModuleGrid from './components/ModuleGrid'
import CLICommands from './components/CLICommands'
import TechStack from './components/TechStack'
import Footer from './components/Footer'

export default function App() {
  return (
    <div>
      <Hero />
      <StatsBar />
      <ModuleGrid />
      <CLICommands />
      <TechStack />
      <Footer />
    </div>
  )
}
