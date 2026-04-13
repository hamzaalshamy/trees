import { Routes, Route } from 'react-router-dom'
import TaxonomyMenu from './TaxonomyMenu'
import RandomForestViz from './RandomForestViz'
import About from './About'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<TaxonomyMenu />} />
      <Route path="/decision-tree" element={<RandomForestViz mode="decision-tree" />} />
      <Route path="/bagging"       element={<RandomForestViz mode="bagging" />} />
      <Route path="/random-forest" element={<RandomForestViz mode="random-forest" />} />
      <Route path="/about"         element={<About />} />
    </Routes>
  )
}
