{{- define "previ-r2d2.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "previ-r2d2.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "previ-r2d2.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "previ-r2d2.labels" -}}
app.kubernetes.io/name: {{ include "previ-r2d2.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}

{{- define "previ-r2d2.selectorLabels" -}}
app.kubernetes.io/name: {{ include "previ-r2d2.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
