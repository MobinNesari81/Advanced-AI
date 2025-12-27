import util
import submission
from mapUtil import createStanfordMap, locationFromTag, makeTag
from grader import extractPath, printPath

cityMap = createStanfordMap()
startLocation = locationFromTag(makeTag("landmark", "oval"), cityMap)
endTag = makeTag("landmark", "bookstore")

ucs = util.UniformCostSearch(verbose=0)
ucs.solve(submission.ShortestPathProblem(startLocation, endTag, cityMap))
path = extractPath(startLocation, ucs)

printPath(path, waypointTags=[], cityMap=cityMap, outPath="path.json")
