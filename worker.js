// Cloudflare Worker relay for bus-map.html (for static hosts with no PHP). Deploy it, then set
// FEED in bus-map.html to the worker's URL. Fixed URL, so it can't be used as an open proxy.
export default {
  async fetch() {
    const r = await fetch("https://metrolinx.tmix.se/gtfs-realtime-saultstemarie/vehiclepositions.pb",
                          {cf: {cacheTtl: 10}});
    return new Response(r.body, {status: r.ok ? 200 : 502, headers: {
      "Content-Type": "application/x-protobuf", "Cache-Control": "max-age=10",
      "Access-Control-Allow-Origin": "*"}});
  },
};
